from SparrowV1_1.core import Sparrow, str2bool
import argparse
import pygame
import torch

'''Sparrow-V1.1 Configuration'''
parser = argparse.ArgumentParser()
parser.add_argument('--map_address', type=str, default='train_maps', help='map address: train_maps / test_maps')
parser.add_argument('--device', type=str, default='cuda', help='running device of Sparrow: cuda / cpu')
parser.add_argument('--ld_num', type=int, default=27, help='number of lidar streams in each world')
parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for one group')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--render_mode', type=str, default='human', help='human / rgb_array / None')
parser.add_argument('--render_speed', type=str, default='fast', help='real / fast / slow')
parser.add_argument('--max_ep_steps', type=int, default=1000, help='maximum episodic steps')
parser.add_argument('--AWARD', type=float, default=80, help='reward of reaching target area')
parser.add_argument('--PUNISH', type=float, default=-10, help='reward when collision happens')
parser.add_argument('--STEP', type=float, default=0.0, help='reward of each step')
parser.add_argument('--normalization', type=str2bool, default=False, help='whether to normalize the observations')
parser.add_argument('--flip', type=str2bool, default=False, help='whether to expand training maps with fliped maps')
parser.add_argument('--noise', type=str2bool, default=False, help='whether to add noise to the observations')
parser.add_argument('--DR', type=str2bool, default=False, help='whether to use Domain Randomization')
parser.add_argument('--DR_freq', type=int, default=int(5e3), help='frequency of Domain Randomization, in total steps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to torch.compile to boost simulation speed')
opt = parser.parse_args()
opt.grouped_ld_num = int(opt.ld_num/opt.ld_GN)
opt.state_dim = 5+opt.grouped_ld_num # [dx,dy,orientation,v_linear,v_angular] + [lidar result]


def main():
    envs = Sparrow(**vars(opt))
    envs.reset()
    while True:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: a = 0
        elif keys[pygame.K_UP]: a = 2
        elif keys[pygame.K_RIGHT]: a = 4
        else: a = 5

        a = torch.ones(envs.N, dtype=torch.long) * a
        s, r, dw, tr, info = envs.step(a)


if __name__ == '__main__':
    main()