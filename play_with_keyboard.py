from SparrowV1_0.core import Sparrow, str2bool
import argparse
import pygame
import torch

def str2bool(v):
    '''transfer str to bool for argparse
    You can just ignore this funciton. It's not related to the RL and Sparrow.'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting For Sparrow'''
parser = argparse.ArgumentParser()
parser.add_argument('--map_address', type=str, default='train_maps', help='map address')
parser.add_argument('--device', type=str, default='cuda:0', help='running devices of sparrow')
parser.add_argument('--ld_num', type=int, default=27, help='number of lidar streams in each world')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--render_mode', type=str, default='human', help='human / rgb_array / None')
parser.add_argument('--render_speed', type=str, default='fast', help='real / fast / slow')
parser.add_argument('--max_ep_steps', type=int, default=1000, help='maximum episodic steps')
parser.add_argument('--AWARD', type=int, default=100, help='reward of reaching target area')
parser.add_argument('--normalization', type=str2bool, default=True, help='whether normalize the observations')
parser.add_argument('--flip', type=str2bool, default=False, help='whether expand training maps with fliped maps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to torch.compile to boost simulation speed')
params = parser.parse_args()


def main():
    envs = Sparrow(params)
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
