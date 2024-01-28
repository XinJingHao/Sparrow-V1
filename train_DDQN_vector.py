from SparrowV1_0.core import Sparrow, str2bool
import copy
import torch
import torch.nn as nn
import numpy as np
import os, shutil
import argparse
from datetime import datetime
import torch.nn.functional as F



'''Hyperparameter Setting for DRL'''
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--write', type=str2bool, default=False, help='Whether use SummaryWriter to record the training curve')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Whether load pretrained model')
parser.add_argument('--ModelIdex', type=int, default=10, help='which model(e.g. 10k.pth) to load')
parser.add_argument('--device', type=str, default='cuda:0', help='device for DDQN, Buffer, and Sparrow')

parser.add_argument('--Max_train_steps', type=int, default=int(5e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e3), help='Model saving interval, in Vsteps.')
parser.add_argument('--random_steps', type=int, default=int(1E4), help='steps for random policy to explore')
parser.add_argument('--init_explore_frac', type=float, default=1.0, help='init explore fraction')
parser.add_argument('--end_explore_frac', type=float, default=0.2, help='end explore fraction')
parser.add_argument('--decay_step', type=int, default=int(40e3), help='linear decay steps(total) for e-greedy noise')
parser.add_argument('--min_eps', type=float, default=0.05, help='minimal e-greedy noise')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--soft_target', type=str2bool, default=False, help='Target net update mechanism')


'''Hyperparameter Setting For Sparrow'''
parser.add_argument('--map_address', type=str, default='same_maps', help='map address')
parser.add_argument('--ld_num', type=int, default=27, help='number of lidar streams in each world')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--render_mode', type=str, default=None, help='human / rgb_array / None')
parser.add_argument('--render_speed', type=str, default='fast', help='real / fast / slow')
parser.add_argument('--max_ep_steps', type=int, default=1000, help='maximum episodic steps')
parser.add_argument('--AWARD', type=float, default=80, help='reward of reaching target area')
parser.add_argument('--PUNISH', type=float, default=-10, help='reward when collision happens')
parser.add_argument('--normalization', type=str2bool, default=True, help='whether normalize the observations')
parser.add_argument('--flip', type=str2bool, default=False, help='whether expand training maps with fliped maps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to torch.compile to boost simulation speed')
opt = parser.parse_args()


opt.actor_envs = len(os.listdir(os.getcwd() + '/SparrowV1_0/' + opt.map_address))
if opt.flip: opt.actor_envs*=2
if opt.write: from torch.utils.tensorboard import SummaryWriter
device = torch.device(opt.device)
opt.state_dim = 5+27 # [dx,dy,orientation,v_linear,v_angular] + [lidar result]
opt.action_dim = 5
opt.buffersize = min(int(1E6), opt.Max_train_steps)


def main(opt):
	# init DDQN model
	if not os.path.exists('model'): os.mkdir('model')
	model = DDQN_Agent(opt)
	if opt.Loadmodel: model.load(opt.ModelIdex)

	if opt.render: # render with Sparrow
		opt.render_mode = 'human'
		eval_envs = Sparrow(opt)
		while True:
			ep_r = evaluate_policy(eval_envs, model, deterministic=True)
			print('Score:', ep_r, '\n')

	else: # trian with Sparrow
		if opt.write:
			# use SummaryWriter to record the training curve
			timenow = str(datetime.now())[0:-10]
			timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
			writepath = 'runs/SparrowV1' + timenow
			if os.path.exists(writepath): shutil.rmtree(writepath)
			writer = SummaryWriter(log_dir=writepath)

		# build vectorized environment and experience replay buffer
		buffer = ReplayBuffer(opt)
		envs = Sparrow(opt)

		s, info = envs.reset() # vectorized env has auto truncate mechanism, so we only reset() once.
		total_steps = 0
		ct = torch.ones(opt.actor_envs, device=device, dtype=torch.bool)
		dones, train_arrival_rate = torch.zeros(opt.actor_envs, dtype=torch.bool, device=device), 0
		while total_steps < opt.Max_train_steps:
			if total_steps < opt.random_steps:
				a = torch.randint(0,opt.action_dim,(opt.actor_envs,),device=device)
			else:
				a = model.select_action(s, deterministic=False)
			s_next, r, dw, tr, info = envs.step(a) #dw(terminated): die or win; tr: truncated
			buffer.add(s, a, r, dw, ct) #注意ct是用上一次step的， 即buffer.add()要在ct = ~(dw + tr)前
			ct = ~(dw + tr)  # 如果当前s_next是”截断状态“或”终止状态“，则s_next与s_next_next是不consistent的，训练时要丢掉
			s = s_next
			total_steps += opt.actor_envs

			# log and record
			train_arrival_rate += (~dones * (r == opt.AWARD)).sum()  # 获得AWARD奖励时，表示抵达终点; Use last done
			dones += (dw + tr)
			if dones.all():
				train_arrival_rate = round(train_arrival_rate.item() / opt.actor_envs, 2)
				if opt.write: writer.add_scalar('Arrival Rate', train_arrival_rate, global_step=total_steps)
				print('Vectorized Sparrow-v1: N:',opt.actor_envs, 'steps: {}k'.format(round(total_steps / 1000,2)), 'Arrival Rate:', train_arrival_rate)
				dones, train_arrival_rate = torch.zeros(opt.actor_envs, dtype=torch.bool, device=device), 0

			# train and fresh e-greedy noise
			if total_steps >= opt.random_steps:
				for _ in range(opt.actor_envs):
					model.train(buffer)
				# fresh vectorized e-greedy noise
				if total_steps % (100*opt.actor_envs) == 0:
					model.fresh_explore_prob(total_steps-opt.random_steps)

			# save model
			if total_steps % (opt.save_interval*opt.actor_envs) == 0:
				model.save(int(total_steps/1e3))



class ReplayBuffer():
	'''Experience replay buffer(For vector env)'''
	def __init__(self, opt):
		self.device = device
		self.max_size = int(opt.buffersize/opt.actor_envs)
		self.state_dim = opt.state_dim
		self.actor_envs = opt.actor_envs
		self.ptr = 0
		self.size = 0
		self.full = False
		self.batch_size = opt.batch_size

		self.s = torch.zeros((self.max_size, opt.actor_envs, opt.state_dim), device=self.device)
		self.a = torch.zeros((self.max_size, opt.actor_envs, 1), dtype=torch.int64, device=self.device)
		self.r = torch.zeros((self.max_size, opt.actor_envs, 1), device=self.device)
		self.dw = torch.zeros((self.max_size, opt.actor_envs, 1), dtype=torch.bool, device=self.device)
		self.ct = torch.zeros((self.max_size, opt.actor_envs, 1),dtype=torch.bool, device=self.device)

	def add(self, s, a, r, dw, ct):
		'''add transitions to buffer'''
		self.s[self.ptr] = s
		self.a[self.ptr] = a.unsqueeze(-1)  #(actor_envs,) to (actor_envs,1)
		self.r[self.ptr] = r.unsqueeze(-1)
		self.dw[self.ptr] = dw.unsqueeze(-1)
		self.ct[self.ptr] = ct.unsqueeze(-1)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		if self.size == self.max_size:
			self.full = True

	def sample(self):
		'''sample batch transitions'''
		if not self.full:
			ind = torch.randint(low=0, high=self.ptr - 1, size=(self.batch_size,), device=self.device)  # sample from [0, ptr-2]
		else:
			ind = torch.randint(low=0, high=self.size - 1, size=(self.batch_size,), device=self.device)  # sample from [0, size-2]
			if self.ptr - 1 in ind: ind = ind[ind != (self.ptr - 1)] # delate ptr - 1 in [0, size-2]

		env_ind = torch.randint(low=0, high=self.actor_envs, size=(len(ind),), device=self.device) # [l,h)
		# [b, s_dim], #[b, 1], [b, 1], [b, s_dim], [b, 1], [b, 1]
		return (self.s[ind,env_ind,:], self.a[ind,env_ind,:],self.r[ind,env_ind,:],
				self.s[ind + 1,env_ind,:], self.dw[ind,env_ind,:], self.ct[ind, env_ind,:])

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
				return a #
			else:
				replace = torch.rand(self.actor_envs, device=device) < self.p  # [n]
				rd_a = torch.randint(0, self.action_dim, (self.actor_envs,), device=device)
				a[replace] = rd_a[replace]
				return a

	def train(self,replay_buffer):
		self.train_counter += 1
		s, a, r, s_next, dw, ct = replay_buffer.sample()

		# Compute the target Q value with Double Q-learning
		with torch.no_grad():
			argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
			max_q_next = self.q_target(s_next).gather(1,argmax_a)
			target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

		# Get current Q estimates
		current_q_a = self.q_net(s).gather(1,a)

		# Mse regression
		if ct.all():
			q_loss = F.mse_loss(current_q_a, target_Q)
		else:
			# discard truncated s, because we didn't save its next state
			q_loss = torch.square(ct * (current_q_a - target_Q)).mean()
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 40)
		self.q_net_optimizer.step()

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


def evaluate_policy(envs, model, deterministic):
	s, info = envs.reset()
	num_env = s.shape[0]
	dones, total_r = torch.zeros(num_env, dtype=torch.bool ,device=device), 0
	while not dones.all():
		a = model.select_action(s, deterministic)
		s, r, dw, tr, info = envs.step(a)
		total_r += (~dones * r).sum()  # use last dones

		dones += (dw + tr)
	return round(total_r.item() / num_env, 2)


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
