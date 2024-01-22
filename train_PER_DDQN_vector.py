from SparrowV1_1.core import Sparrow, str2bool
import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
import os, shutil
import argparse
import torch
import copy


'''Hyperparameter Setting for DRL'''
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--write', type=str2bool, default=False, help='Whether use SummaryWriter to record the training curve')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Whether load pretrained model')
parser.add_argument('--ModelIdex', type=int, default=10, help='which model(e.g. 10k.pth) to load')

parser.add_argument('--Max_train_steps', type=int, default=int(5E5), help='Max training steps')
parser.add_argument('--eval_freq', type=int, default=int(2E2), help='evaluation frequency, in Vsteps')
parser.add_argument('--init_explore_frac', type=float, default=1.0, help='init explore fraction')
parser.add_argument('--end_explore_frac', type=float, default=0.2, help='end explore fraction')
parser.add_argument('--decay_step', type=int, default=int(100E3), help='linear decay steps(total) for e-greedy noise')
parser.add_argument('--min_eps', type=float, default=0.05, help='minimal e-greedy noise')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--soft_target', type=str2bool, default=False, help='Target net update mechanism')
parser.add_argument('--alpha', type=float, default=0.4, help='alpha of PER')
parser.add_argument('--beta', type=float, default=0.6, help='beta of PER')


'''Hyperparameter Setting For Sparrow'''
parser.add_argument('--map_address', type=str, default='train_maps', help='map address: train_maps / test_maps')
parser.add_argument('--device', type=str, default='cuda', help='running device of Sparrow: cuda / cpu')
parser.add_argument('--ld_num', type=int, default=27, help='number of lidar streams in each world')
parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for one group')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--render_mode', type=str, default=None, help='human / rgb_array / None')
parser.add_argument('--render_speed', type=str, default='fast', help='real / fast / slow')
parser.add_argument('--max_ep_steps', type=int, default=1000, help='maximum episodic steps')
parser.add_argument('--AWARD', type=float, default=80, help='reward of reaching target area')
parser.add_argument('--PUNISH', type=float, default=-10, help='reward when collision happens')
parser.add_argument('--STEP', type=float, default=0.0, help='reward of each step')
parser.add_argument('--normalization', type=str2bool, default=True, help='whether to normalize the observations')
parser.add_argument('--flip', type=str2bool, default=True, help='whether to expand training maps with fliped maps')
parser.add_argument('--noise', type=str2bool, default=False, help='whether to add noise to the observations')
parser.add_argument('--DR', type=str2bool, default=False, help='whether to use Domain Randomization')
parser.add_argument('--DR_freq', type=int, default=int(5e3), help='frequency of Domain Randomization, in total steps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to torch.compile to boost simulation speed')
opt = parser.parse_args()

opt.actor_envs = len(os.listdir(os.getcwd() + '/SparrowV1_1/' + opt.map_address))
if opt.flip: opt.actor_envs*=2
if opt.write: from torch.utils.tensorboard import SummaryWriter
device = torch.device(opt.device)
assert opt.ld_num % opt.ld_GN == 0 #ld_num must be divisible by ld_GN
opt.grouped_ld_num = int(opt.ld_num/opt.ld_GN)
opt.state_dim = 5+opt.grouped_ld_num # [dx,dy,orientation,v_linear,v_angular] + [lidar result]
opt.action_dim = 5
opt.buffersize = min(int(1E6), opt.Max_train_steps)
# print(opt)


def main(opt):
	# init DDQN model
	if not os.path.exists('model'): os.mkdir('model')
	model = DDQN_Agent(opt)
	if opt.Loadmodel: model.load(opt.ModelIdex)

	if opt.render: # render with Sparrow
		opt.render_mode = 'human'
		eval_envs = Sparrow(**vars(opt))
		while True:
			ep_r, arrival_rate = evaluate_policy(eval_envs, model, deterministic=True, AWARD=opt.AWARD)
			print(f'Averaged Score:{ep_r} \n Arrival rate:{arrival_rate} \n')

	else: # trian with Sparrow
		if opt.write:
			# use SummaryWriter to record the training curve
			timenow = str(datetime.now())[0:-10]
			timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
			writepath = f'runs/SparrowV1.1_PER_DDQN' + timenow
			if os.path.exists(writepath): shutil.rmtree(writepath)
			writer = SummaryWriter(log_dir=writepath)

		# build vectorized environment and experience replay buffer
		buffer = PrioritizedReplayBuffer(opt)
		envs = Sparrow(**vars(opt))

		# build train/test env for evaluation
		eval_confg = copy.deepcopy(opt)
		eval_confg.flip = False
		eval_confg.map_address = 'train_maps'
		train_envs = Sparrow(**vars(eval_confg))
		eval_confg.map_address = 'test_maps'
		test_envs = Sparrow(**vars(eval_confg))


		s, info = envs.reset() # vectorized env has auto truncate mechanism, so we only reset() once.
		total_steps = 0
		ct = torch.ones(opt.actor_envs, device=device, dtype=torch.bool)
		while total_steps < opt.Max_train_steps:
			a = model.select_action(s, deterministic=False) # will generate random steps at the beginning
			s_next, r, dw, tr, info = envs.step(a) #dw(terminated): die or win; tr: truncated
			buffer.add(s, a, r, dw, ct) #注意ct是用上一次step的， 即buffer.add()要在ct = ~(dw + tr)前； (s1,a1,r2,dw2,ct1)
			ct = ~(dw + tr)  # 如果当前s_next是”截断状态“或”终止状态“，则s_next与s_next_next是不consistent的，训练时要丢掉
			s = s_next
			total_steps += opt.actor_envs

			'''train and fresh e-greedy noise'''
			if total_steps >= 2E4:
				for _ in range(opt.actor_envs):
					model.train(buffer)
				# fresh vectorized e-greedy noise
				if total_steps % (100*opt.actor_envs) == 0:
					model.fresh_explore_prob(total_steps)

			'''evaluate, log, record, save'''
			if total_steps % (opt.eval_freq*opt.actor_envs) == 0:
				_, train_arrival_rate = evaluate_policy(train_envs, model, deterministic=True, AWARD=opt.AWARD)
				_, test_arrival_rate = evaluate_policy(test_envs, model, deterministic=True, AWARD=opt.AWARD)
				print(f'Sparrow-v1  N:{opt.actor_envs};  Total steps: {round(total_steps / 1e3, 2)}k;  Train Arrival rate:{train_arrival_rate};  Test Arrival rate:{test_arrival_rate}')
				if opt.write:
					writer.add_scalar('train_arrival_rate', train_arrival_rate, global_step=total_steps)
					writer.add_scalar('test_arrival_rate', test_arrival_rate, global_step=total_steps)
				if test_arrival_rate > 0.7: model.save(int(total_steps / 1e3))

		envs.close()
		train_envs.close()
		test_envs.close()

def evaluate_policy(envs, model, deterministic, AWARD):
	s, info = envs.reset()
	num_env = s.shape[0]
	dones, total_r, arrival_rate = torch.zeros(num_env, dtype=torch.bool ,device=device), 0, 0
	while not dones.all():
		a = model.select_action(s, deterministic)
		s, r, dw, tr, info = envs.step(a)
		total_r += (~dones * r).sum()  # use last dones
		arrival_rate += (~dones * (r==AWARD)).sum() # 获得AWARD奖励时，表示抵达终点

		dones += (dw + tr)
	return round(total_r.item() / num_env, 2), round(arrival_rate.item() / num_env, 2)

class PrioritizedReplayBuffer():
	'''Experience replay buffer(For vector env)'''
	def __init__(self, opt):
		self.device = device
		self.max_size = int(opt.buffersize/opt.actor_envs)
		self.state_dim = opt.state_dim
		self.actor_envs = opt.actor_envs
		self.ptr = 0
		self.size = 0
		self.full = False
		self.bs_per_env = int(opt.batch_size/self.actor_envs) # batchsize per env: 每次采样时，每个env出多少个数据
		self.constant_env_idx = torch.arange(opt.actor_envs).unsqueeze(-1).repeat((1, self.bs_per_env)).view(-1)
		# eg: actor_envs=2, bs_per_env=3, then constant_env_idx=[0,0,0,1,1,1]

		self.s = torch.zeros((self.max_size, opt.actor_envs, opt.state_dim), device=self.device)
		self.a = torch.zeros((self.max_size, opt.actor_envs, 1), dtype=torch.int64, device=self.device)
		self.r = torch.zeros((self.max_size, opt.actor_envs, 1), device=self.device)
		self.dw = torch.zeros((self.max_size, opt.actor_envs, 1), dtype=torch.bool, device=self.device)
		self.priorities = torch.zeros((self.max_size, opt.actor_envs), device=self.device)

		self.max_p = 1.0
		self.alpha = opt.alpha
		self.beta = opt.beta


	def add(self, s, a, r, dw, ct):
		'''Add transitions to buffer
		   ct[i] 表示 s[i]与s[i+1]是否来自同一个回合'''
		self.s[self.ptr] = s
		self.a[self.ptr] = a.unsqueeze(-1)  #(actor_envs,) to (actor_envs,1)
		self.r[self.ptr] = r.unsqueeze(-1)
		self.dw[self.ptr] = dw.unsqueeze(-1)
		self.priorities[self.ptr] = ct*self.max_p # s[t]和s[t+1]不连续时，采样概率为0，永远无法被sample; (actor_envs,)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		if self.size == self.max_size:
			self.full = True

	def sample(self):
		'''sample batch transitions'''
		# 因为没有state[size]，所以从[0, size-1)中sample：
		Prob_torch_gpu = self.priorities[0: self.size - 1].clone()  # 这里必须clone
		# 因为state[ptr-1]和state[ptr]不来自同一个episode， 所以采样时不能采到ptr-1:
		if self.ptr < self.size: Prob_torch_gpu[self.ptr - 1].fill_(0.)

		step_idx = torch.multinomial(Prob_torch_gpu.T, num_samples=self.bs_per_env, replacement=True).view(-1) # (batchsize,)
		IS_weight = (self.size * Prob_torch_gpu[step_idx, self.constant_env_idx]) ** (-self.beta) # (batchsize,)
		Normed_IS_weight = (IS_weight / IS_weight.max()).unsqueeze(-1)  # (batchsize,1)

		return (self.s[step_idx, self.constant_env_idx,:],  # [b, s_dim]
				self.a[step_idx, self.constant_env_idx,:],  # [b, 1]
				self.r[step_idx, self.constant_env_idx,:],  # [b, 1]
				self.s[step_idx+1, self.constant_env_idx,:],  # [b, s_dim]
				self.dw[step_idx, self.constant_env_idx,:],  # [b, 1]
				step_idx, # (batchsize,)
				Normed_IS_weight) # (batchsize,1)



def orthogonal_init(layer, gain=1.414):
	for name, param in layer.named_parameters():
		if 'bias' in name:
			nn.init.constant_(param, 0)
		elif 'weight' in name:
			nn.init.orthogonal_(param, gain=gain)
	return layer

def build_net(layer_shape, activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [orthogonal_init(nn.Linear(layer_shape[j], layer_shape[j+1])), act()]
	return nn.Sequential(*layers)

class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.Q = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q = self.Q(s)
		return q

class DDQN_Agent(object):
	def __init__(self,opt):
		self.q_net = Q_Net(opt.state_dim, opt.action_dim, [opt.net_width,int(opt.net_width/2)]).to(device)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.actor_envs = opt.actor_envs
		self.action_dim = opt.action_dim
		self.gamma = opt.gamma
		self.tau = 0.005
		self.soft_target = opt.soft_target
		self.train_counter = 0

		# vectorized e-greedy exploration
		self.explore_frac_scheduler = LinearSchedule(opt.decay_step, opt.init_explore_frac, opt.end_explore_frac)
		self.p = torch.ones(opt.actor_envs, device=device)
		self.min_eps = opt.min_eps

	def fresh_explore_prob(self, steps):
		#fresh vectorized e-greedy noise
		explore_frac = self.explore_frac_scheduler.value(steps)
		i = int(explore_frac * self.actor_envs)
		explore = torch.arange(i, device=device) / (2 * i)  # 0 ~ 0.5

		self.p.fill_(self.min_eps)
		self.p[self.actor_envs - i:] += explore
		self.p = self.p[torch.randperm(self.actor_envs)]  # 打乱vectorized e-greedy noise, 让探索覆盖每一个地图

	def select_action(self, s, deterministic):
		'''Input: batched state, (n,32), torch.tensor, on device
		   Output: batched action, (n,), torch.tensor, on device '''
		with torch.no_grad():
			a = self.q_net(s).argmax(dim=-1)
			if deterministic:
				return a
			else:
				replace = torch.rand(self.actor_envs, device=device) < self.p  # [n]
				rd_a = torch.randint(0, self.action_dim, (self.actor_envs,), device=device)
				a[replace] = rd_a[replace]
				return a

	def train(self,replay_buffer):
		self.train_counter += 1
		s, a, r, s_next, dw, step_idx, Normed_IS_weight = replay_buffer.sample()

		# Compute the target Q value with Double Q-learning
		with torch.no_grad():
			argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
			max_q_next = self.q_target(s_next).gather(1,argmax_a)
			target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

		# Get current Q estimates
		current_q_a = self.q_net(s).gather(1,a)

		# Mse regression
		q_loss = torch.square(Normed_IS_weight * (target_Q - current_q_a)).mean()
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
		self.q_net_optimizer.step()

		# update priorites of the current batch
		with torch.no_grad():
			batch_priorities = ((torch.abs(target_Q - current_q_a) + 0.01)**replay_buffer.alpha).squeeze(-1) #(batchsize,) on devive
			replay_buffer.priorities[step_idx, replay_buffer.constant_env_idx] = batch_priorities
			current_max_p = batch_priorities.max()
			if current_max_p > replay_buffer.max_p: replay_buffer.max_p = current_max_p # 更新最大priority

		# Update the target net
		if self.soft_target:
			# soft target update
			for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else:
			# hard target update
			if self.train_counter % int(1/self.tau):
				for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
					target_param.data.copy_(param.data)

	def save(self,steps):
		torch.save(self.q_net.state_dict(), "./model/{}k.pth".format(steps))

	def load(self,steps):
		self.q_net.load_state_dict(torch.load("./model/{}k.pth".format(steps)))
		self.q_target = copy.deepcopy(self.q_net)
		for p in self.q_target.parameters(): p.requires_grad = False


class LinearSchedule(object):
	def __init__(self, schedule_timesteps, initial_p, final_p):
		"""Linear interpolation between initial_p and final_p over"""
		self.schedule_timesteps = schedule_timesteps
		self.initial_p = initial_p
		self.final_p = final_p

	def value(self, t):
		fraction = min(float(t) / self.schedule_timesteps, 1.0)
		return self.initial_p + fraction * (self.final_p - self.initial_p)

if __name__ == '__main__':
	main(opt)
