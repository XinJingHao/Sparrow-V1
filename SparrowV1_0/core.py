from collections import deque
import numpy as np
import torch
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

'''Recommended Hyperparameter Setting For Sparrow:
from SparrowV1_0.core import Sparrow, str2bool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--map_address', type=str, default='train_maps', help='map address')
parser.add_argument('--device', type=str, default='cuda:0', help='running device of sparrow')
parser.add_argument('--ld_num', type=int, default=27, help='number of lidar streams in each world')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of the world that be rendered')
parser.add_argument('--render_mode', type=str, default='human', help='human / rgb_array / None')
parser.add_argument('--render_speed', type=str, default='fast', help='real / fast / slow')
parser.add_argument('--max_ep_steps', type=int, default=1000, help='maximum episodic steps')
parser.add_argument('--AWARD', type=int, default=100, help='reward of reaching target area')
parser.add_argument('--normalization', type=str2bool, default=True, help='whether normalize the observations')
parser.add_argument('--flip', type=str2bool, default=True, help='whether expand training maps with flipped maps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to torch.compile to boost simulation speed')
params = parser.parse_args()
'''




'''
Sparrow-V1.0: A Reinforcement Learning Friendly Simulator for Mobile Robot

Several good features:
· Vectorizable (Enable super fast data collection)
· Lightweight (Consume only 140~300 mb GPU memories for vectorized environments)
· Standard Gym API with Pytorch data flow
· GPU/CPU are both acceptable (If you use Pytorch to build your RL model, you can run your RL model and Sparrow both on GPU. Then you don't need to transfer the data from CPU to GPU anymore.)
· Easy to use (20kb pure Python files. Just import, never worry about installation)
· Ubuntu/Windows/MacOS are all supported
· Accept image as map (Customize your own environments easily and rapidly)
· Detailed comments on source code

Only for non-commercial purposes.
All rights reserved. 

“The sparrow may be small but it has all the vital organs.”
Developed by Jinghao Xin. Github：https://github.com/XinJingHao


Current version: V1.0 (2023/6/23)
'''

# Color of obstacles when render
_OBS = (64, 64, 64)

class Sparrow():
    '''Sparrow-V1.0'''
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, params):
        self.window_size = 366  # 366.0cm = 366个栅格，索引[0,365]
        self.state_dim = 5+params.ld_num
        self.map_address = params.map_address
        self.dvc = torch.device(params.device)# running device of Sparrow, better use GPU to accelerate simulation
        self.ld_num = params.ld_num  # number of lidar streams in each world
        self.ri = params.ri # render index
        self.render_mode = params.render_mode # should be one of "human", "rgb_array", None
        self.render_speed = params.render_speed # should be one of 'real', 'fast', 'slow'
        self.max_ep_steps = params.max_ep_steps  # maximum episodic steps
        self.AWARD = params.AWARD  # 通关奖励
        self.normalization = params.normalization
        self.flip = params.flip # Expand the training maps by flipping the maps
        self.compile = params.compile # use torch.compile to boost simulation speed

        '''Map initialization'''
        # ['map1.png' 'map10.png' 'map11.png' ... 'map14.png' 'map15.png' 'map2.png' 'map3.png' ... 'map9.png']
        self.maps = np.sort(os.listdir(os.getcwd() + '/SparrowV1_0/' + self.map_address))
        self.N = len(self.maps) # Number of vectorized Env, equating to number of maps
        if self.flip: self.N *= 2 # 翻转后训练数据数量翻倍
        self._bound_init() # 为所有地图生成bound_code, 补齐后整合为self.vec_bound_code
        self.target_area = 100 # # enter the target_area×target_area (cm) space in the upper left of the map will be considered win.

        '''Car initialization'''
        self.car_radius = 9  # cm
        self.collision_trsd = self.car_radius + 5  # collision threshould, in cm
        self.ctrl_interval = 0.1 # control interval, in second
        self.ctrl_delay = 1 # in ctrl_interval
        self.ctrl_pipe = deque(maxlen=self.ctrl_delay+1) # holding the delayed action
        self.K = 0.6
        self.v_linear_max = 18 # max linear velocity, in cm/s
        self.v_angular_max = 1 # max angular velocity, in rad/s
        self.a_space = torch.tensor([[0.2*self.v_linear_max , self.v_angular_max],[self.v_linear_max , self.v_angular_max],
                                     [self.v_linear_max, 0], # v_linear, v_angular
                                     [self.v_linear_max, -self.v_angular_max],[0.2*self.v_linear_max, -self.v_angular_max],
                                     [0,0]], device=self.dvc) # a_space[-1] is no_op，agent is not allowed to use.

        '''Lidar initialization'''
        self.ld_acc = 3  # lidar scan accuracy (cm). Reducing accuracy can accelerate simulation;
        self.ld_range = 100  # max scanning distance of lidar (cm). Reducing ld_range can accelerate simulation;
        self.ld_scan_result = torch.zeros((self.N, self.ld_num), device=self.dvc)  # used to hold lidar scan result, (N, ld_num)
        self.ld_angle_interval = torch.arange(self.ld_num, device=self.dvc) * 1.5 * torch.pi / (self.ld_num - 1) - 0.75 * torch.pi #(ld_num, )
        self.ld_angle_interval = self.ld_angle_interval.unsqueeze(dim=0).repeat((self.N, 1)) # (N, ld_num)

        '''Pygame initialization'''
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        # "human": will render in a pygame window
        # "rgb_array" : calling env.render() will return the airscape of the robot in numpy.ndarray
        # None: not render anything
        self.window = None
        self.clock = None
        self.canvas = None

        '''Internal variables initialization'''
        # 初始化变量地址, 防止使用时再开辟cuda内存, 节省时间
        self.car_state = torch.zeros((self.N, 5), device=self.dvc, dtype=torch.float32)
        self.step_counter_vec = torch.zeros(self.N, dtype=torch.long, device=self.dvc)
        self.reward_vec = torch.zeros(self.N, device=self.dvc) # vectorized reward signal
        self.dw_vec = torch.zeros(self.N, dtype=torch.bool, device=self.dvc) # vectorized terminated signal
        self.tr_vec = torch.zeros(self.N, dtype=torch.bool, device=self.dvc)  # vectorized truncated signal
        self.done_vec = torch.zeros(self.N, dtype=torch.bool, device=self.dvc)  # vectorized done signal
        self.state_upperbound = torch.ones(self.state_dim, device=self.dvc) # used for state normalization
        self.state_upperbound[0:2] *= self.window_size
        self.state_upperbound[2] *= 1 # 仅用于补位,后面单独归一化
        self.state_upperbound[3] *= self.v_linear_max
        self.state_upperbound[4] *= self.v_angular_max
        self.state_upperbound[5:self.state_dim] *= self.ld_range

        if self.dvc == 'cpu':
            print('Although Sparrow-V1 can be deployed on CPU, we strongly recommend you use GPU to accelerate simulation!')
        else:
            # 编译雷达扫描函数，加速仿真. 但有些显卡上会报错.
            if self.compile == True:
                self._ld_scan_vec = torch.compile(self._ld_scan_vec)
            else:
                print("When instantiate Sparrow-V1, you can set '--compile True' to boost the simulation speed. But this may cause errors on some GPU.")

    def _world_2_grid(self, coordinate_wd):
        ''' Convert world coordinates (denoted by _wd, continuous, unit: cm) to grid coordinates (denoted by _gd, discrete, 1 grid = 1 cm)
            Input: torch.tensor; Output: torch.tensor; Shape: Any shape '''
        return coordinate_wd.floor().int()

    def _bound_init(self):
        '''Load the map_.png, extract the [x,y] of obstacle points, and map them into [x*window_size+y]'''
        max_bound_num = 0
        bound_code_list = []
        if self.flip:
            flip_bound_code_list = []

        # 加载各个地图, 提取bound_gd, 转换为bound_code, 存入bound_code_list中
        for map_name in self.maps: # ['map1.png' 'map10.png' 'map11.png' ... 'map14.png' 'map15.png' 'map2.png' 'map3.png' ... 'map9.png']
            map_pyg = pygame.image.load(os.getcwd() + '/SparrowV1_0/' + self.map_address + '/' + map_name) # 不能用plt.imread读, 有bug
            map_np = pygame.surfarray.array3d(map_pyg)[:, :, 0]

            x_, y_ = np.where(map_np == 0) # 障碍物栅格的x,y坐标
            bound_gd = torch.tensor(np.stack((x_,y_), axis=1)) # 障碍物栅格点, 同Sparrow-v0, (num, 2), on cpu
            bound_code = bound_gd[:, 0] * self.window_size + bound_gd[:, 1] # 二维矩阵编码后的bound_gd, (num, ), on cpu
            bound_code_list.append(bound_code)
            if max_bound_num < bound_code.shape[0]: max_bound_num = bound_code.shape[0] # 所有地图中bound栅格点最大数量,用于后面补齐

            if self.flip: # 计算翻转地图(x,y坐标交换)的编码值
                flip_bound_code = bound_gd[:, 1] * self.window_size + bound_gd[:, 0] # 二维矩阵编码后的flip_bound_gd, (num, ), on cpu
                flip_bound_code_list.append(flip_bound_code)

        if self.flip:
            for fliped_data in flip_bound_code_list:
                bound_code_list.append(fliped_data) # 将flip后的数据添加在最后，防止与ri错位


        # 将各个地图的bound_code统一补齐至max_bound_num, 方便后续广播运算
        self.vec_bound_code = torch.zeros((self.N, max_bound_num), dtype=torch.long) # (N, max_bound_num), on cpu
        for _ in range(self.N):
            conpensate = torch.ones(size=(max_bound_num-len(bound_code_list[_]),), dtype=torch.long)*bound_code_list[_][0] #(max_bound_num-num, ), on cpu
            self.vec_bound_code[_] = torch.cat((bound_code_list[_], conpensate))

        # 拓充维度, 便于后续计算goon: ld_end_gd[:, :, None] - bound_gd[:, None, :]
        self.vec_bound_code = self.vec_bound_code[:, None, :].to(self.dvc)

    def reset(self):
        # 小车位置初始化
        self.car_state *= 0
        self.car_state[:, 0:2] = (316 + (torch.rand((self.N, 2), device=self.dvc) - 0.5) * 20)  # [306~326]
        self.car_state[:, 2] = torch.rand(self.N, device=self.dvc) * 2 * torch.pi

        # 步数初始化
        self.step_counter_vec *= 0
        
        # 控制指令管道初始化: action5:[0,0]
        for i in range(self.ctrl_delay+1):
            self.ctrl_pipe.append(torch.ones(self.N, dtype=torch.long, device=self.dvc)*(len(self.a_space)-1))

        # 获取初始状态
        observation_vec, info = self._get_obs(), None
        if self.done_vec.any(): raise RuntimeError('Done Signal in Reset State!')

        # Render
        if self.render_mode == "human": self._render_frame()

        return observation_vec, info

    def _ld_not_in_bound_vec(self):
        '''Check whether ld_end_code is not in bound_code in a vectorized way => goon'''
        # 将ld_end_gd中的每个元素与bound_gd相减
        pre_goon = self.ld_end_code[:, :, None] - self.vec_bound_code # vec_bound_code 在生成时已经加过维度了

        # 判断是否存在零值，存在即A中的元素在B中存在
        return ~torch.any(pre_goon == 0, dim=2) # goon

    def _ld_scan_vec(self):
        '''Get the scan result (in vectorized worlds) of lidars. '''
        # 扫描前首先同步雷达与小车位置:
        self.ld_angle = self.ld_angle_interval + self.car_state[:,2,None]# 雷达-小车方向同步, (N, ld_num) + (N, 1) = (N, ld_num)
        self.ld_vectors_wd = torch.stack((torch.cos(self.ld_angle), -torch.sin(self.ld_angle)), dim=2)  # 雷达射线方向, (N,ld_num,2), 注意在unified_cs中是-sin
        self.ld_end_wd = self.car_state[:,None,0:2] + self.car_radius * self.ld_vectors_wd  # 扫描过程中，雷达射线末端世界坐标(初始化于小车轮廓), (N,1,2)+(N,ld_num,2)=(N,ld_num,2)
        self.ld_end_gd = self._world_2_grid(self.ld_end_wd)  # 扫描过程中，雷达射线末端栅格坐标, (N,ld_num,2)
        self.ld_end_code = self.ld_end_gd[:,:,0]*self.window_size + self.ld_end_gd[:,:,1]# 扫描过程中，雷达射线末端栅格坐标的编码值, (N,ld_num)

        # 扫描初始化
        self.ld_scan_result *= 0  # 结果归零, (N, ld_num)
        increment = self.ld_vectors_wd * self.ld_acc  # 每次射出的增量, (N,ld_num,2)

        # 并行式烟花式扫描
        for i in range( int((self.ld_range-self.car_radius)/self.ld_acc) + 2 ): # 多扫2次，让最大值超过self.ld_range，便于clamp
            # 更新雷达末端位置
            goon = self._ld_not_in_bound_vec() # 计算哪些ld_end_code不在bound_code里, 即还没有扫到障碍 #(N, ld_num)
            self.ld_end_wd += (goon[:,:,None] * increment)  # 更新雷达末端世界坐标,每次射 ld_acc cm #(N, ld_num,1)*(N,ld_num,2)=(N,ld_num,2)
            self.ld_end_gd = self._world_2_grid(self.ld_end_wd)# 更新雷达末端栅格坐标（必须更新，下一轮会调用）, (N,ld_num,2)
            self.ld_end_code = self.ld_end_gd[:, :, 0] * self.window_size + self.ld_end_gd[:, :, 1]# 更新雷达末端栅格坐标编码值, (N,ld_num)
            self.ld_scan_result += (goon * self.ld_acc)# 累计扫描距离 (N, ld_num)

            if (~goon).all(): break # 如果所有ld射线都被挡，则扫描结束

        # 扫描的时候从小车轮廓开始扫的，最后要补偿小车半径的距离; (ld_num, ); torch.tensor
        self.ld_scan_result = (self.ld_scan_result + self.car_radius).clamp(0,self.ld_range) #(N, ld_num)

    def _Sparse_reward_function(self):
        '''Calculate vectorized reward, terminated(dw), truncated(tr), done(dw+tr) signale'''
        self.tr_vec = (self.step_counter_vec > self.max_ep_steps)# truncated signal (N,)

        dead_vec = (self.ld_scan_result < self.collision_trsd).any(dim=-1)  # (N,)
        win_vec = (self.car_state[:,0:2] < self.target_area).all(dim=-1) # (N,)
        self.dw_vec = dead_vec + win_vec # terminated signal (N,)

        self.done_vec = self.tr_vec + self.dw_vec # (N,), used for AutoReset

        self.reward_vec.fill_(-0.05)
        self.reward_vec[win_vec] = self.AWARD
        self.reward_vec[dead_vec] = self.PUNISH

    def _Normalize(self, observation):
        '''Normalize the raw observations (N,32) to relative observations (N,32)'''
        # Normalize the orientation:
        beta = torch.arctan(observation[:,0]/observation[:,1]) + torch.pi/2 # arctan(x/y)+π/2
        observation[:, 2] = (beta - observation[:, 2]) / torch.pi
        observation[:][observation[:, 2]<-1] += 2
        # Normalize other observation:
        return observation/self.state_upperbound

    def _get_obs(self):
        '''Return: UNormalized vectorized [dx, dy, theta, v_linear, v_angular, lidar_results(0), ..., lidar_results(26)] in shape (N,32) '''
        self._ld_scan_vec()  # Get the scan result of lidar, stored in self.ld_scan_result, in shape (N, ld_num)
        self._Sparse_reward_function() # calculate reward, dw, tr, done signals
        observation_vec = torch.concat((self.car_state, self.ld_scan_result), dim=-1) #(N, 5) cat (N, ld_num) = (N, 32)
        if self.normalization:
            observation_vec = self._Normalize(observation_vec)
        return observation_vec

    def _Kinematic_model_vec(self, a):
        ''' V_now = K*V_previous + (1-K)*V_target
            Input: action index, (N,)
            Output: [v_l, v_l, v_a], (N,3)'''
        self.car_state[:,3:5] = self.K * self.car_state[:,3:5] + (1-self.K)*self.a_space[a] # self.a_space[a] is (N,2)
        return torch.stack((self.car_state[:,3],self.car_state[:,3],self.car_state[:,4]),dim=1) # [v_l, v_l, v_a], (N,3)


    def step(self,current_a): # current_a should be vectorized action of dim (N, ) on self.dvc
        self.step_counter_vec += 1

        # control delay mechanism
        a = self.ctrl_pipe.popleft() # a is the delayed action, (N,)
        self.ctrl_pipe.append(current_a) # current_a is the action mapped by the current state

        # calculate and update the velocity of the car based on the delayed action and the Kinematic_model
        velocity = self._Kinematic_model_vec(a) #[v_l, v_l, v_a], (N,3)

        # calculate and update the [dx,dy,orientation] of the car
        self.car_state[:,0:3] += self.ctrl_interval * velocity * torch.stack((torch.cos(self.car_state[:,2]),
                                                                              -torch.sin(self.car_state[:,2]),
                                                                              torch.ones(self.N,device=self.dvc)), dim=1)
        # keep the orientation between [0,2pi]
        self.car_state[:,2] %= (2 * torch.pi)

        # get obervation and RL signals
        observation_vec, info,  = self._get_obs(), None

        if self.render_mode == "human": self._render_frame()

        # reset some of the envs based on the done_vec signal
        self._AutoReset()

        return observation_vec, self.reward_vec.clone(), self.dw_vec.clone(), self.tr_vec.clone(), info

    def _AutoReset(self):
        '''Reset done掉的env（没有done的不受影响）'''
        if self.done_vec.any():
            n_dones = self.done_vec.sum()
            # 1) seset the car pose
            self.car_state[self.done_vec] *= 0
            self.car_state[self.done_vec, 0:2] = (316 + (torch.rand((n_dones,2), device=self.dvc) - 0.5) * 20)  # [306~326]
            self.car_state[self.done_vec, 2] = torch.rand(n_dones, device=self.dvc) * 2 * torch.pi

            # 2) reset the step counter
            self.step_counter_vec[self.done_vec] = 0

            # 3) reset the ctrl_pipe：
            # 单独重置某个环境的ctrl_pipe比较麻烦，并且ctrl_delay较短时是否重置影响不大(类似于增加随机性)
            # 所以这里为了仿真速度，我们不重置ctrl_pipe


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size , self.window_size ))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # init canvas
        if self.canvas is None :
            self.canvas = pygame.Surface((self.window_size , self.window_size ))
            self.map = pygame.image.load(os.getcwd() + '/SparrowV1_0/'+self.map_address+'/' + self.maps[self.ri])

        # draw obstacles on canvas
        self.canvas.blit(self.map, self.canvas.get_rect())

        # draw target area
        pygame.draw.rect(self.canvas, (100, 255, 100),pygame.Rect((7,7), (97, 97)),width=6)

        # draw lidar rays on canvas
        ld_result = self.ld_scan_result[self.ri].cpu().clone() # (ld_num, ), on cpu
        ld_real_sta_gd = self._world_2_grid(self.car_state[self.ri,0:2]).cpu().numpy() #(2,)
        ld_real_end_gd = self._world_2_grid(self.car_state[self.ri,0:2].cpu() + ld_result.unsqueeze(-1) * self.ld_vectors_wd[self.ri].cpu()).numpy() #(ld_num,2)
        for i in range(self.ld_num):
            e = 255*ld_result[i]/self.ld_range
            pygame.draw.line(self.canvas, (255-e, 0, e), ld_real_sta_gd, ld_real_end_gd[i], width=2)

        # draw collision threshold on canvas
        pygame.draw.circle(
            self.canvas,
            _OBS,
            self._world_2_grid(self.car_state[self.ri,0:2]).cpu().numpy(),
            self.collision_trsd,
        )

        # draw robot on canvas
        pygame.draw.circle(
            self.canvas,
            (200, 128, 250),
            self._world_2_grid(self.car_state[self.ri,0:2]).cpu().numpy(),
            self.car_radius,
        )
        # draw robot orientation on canvas
        head = self.car_state[self.ri,0:2].cpu() + self.car_radius * torch.tensor([torch.cos(self.car_state[self.ri,2]), -torch.sin(self.car_state[self.ri,2])])
        pygame.draw.line(
            self.canvas,
            (0, 255, 255),
            self._world_2_grid(self.car_state[self.ri,0:2]).cpu().numpy(),
            self._world_2_grid(head).numpy(),
            width=2
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            if self.render_speed == 'real':
                self.clock.tick(int(1 / self.ctrl_interval))
            elif self.render_speed == 'fast':
                self.clock.tick(0)
            elif self.render_speed == 'slow':
                self.clock.tick(5)
            else:
                print('Wrong Render Speed, only "real"; "fast"; "slow" is acceptable.')

        else: #rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def str2bool(v):
    '''Fix the bool BUG for argparse: transfer str to bool'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1', 'T'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0', 'F'):
        return False
    else:
        print('Wrong Input Type!')
