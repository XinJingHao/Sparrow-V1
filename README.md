
<div align="center">
  <a ><img width="300px" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/LOGO%20sparrow.jpg"></a>
</div>

## Sparrow-V1.0: A Reinforcement Learning Friendly Simulator for Mobile Robot

<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/render.gif">
</div>

![Python](https://img.shields.io/badge/Python-blue)
![DRL](https://img.shields.io/badge/DRL-blueviolet)
![Mobile Robot](https://img.shields.io/badge/MobileRobot-ff69b4)


## What's New in V1.0:

Sparrow-V1.0 is a new-generation mobile robot simulator from the Sparrow family, which puts paramount importance on its simulation **speed and lightness**. The comparison between Sparrow-V0 and Sparrow-V1.0 is shown below. 

<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/architecture.png">
</div>

In Sparrow-V0, the vectorization relies on [gym.vector](https://www.gymlibrary.dev/content/vectorising/), which leads to 2 unavoidable limitations: 1) the core calculation of Sparrow-V0 is separately requested to GPU by different CPU cores, wasting the parallel computation feature of GPU and resulting in heavy GPU memory occupation. 2) the gym.vector only supports data (e.g. the state, action, reward, terminated, truncated signals) in [numpy](https://numpy.org/) format, giving rise to a unfavorable data conversion procedure ( _Sparrow.Variables(gpu) → Gym.Variables(cpu) → DRL.Variables(gpu)_  ) that  dramatically slow down the training speed. 

To tackle these two issues, the Sparrow-V1.0 concatenates all the variables from different worlds (vectorized environments) and feeds them to the GPU together, unleashing the parallel computing power of GPU and bypassing gym.vector so that omits the data conversion. Additionally, with the publication of [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/), the core calculation process of Sparrow, the LiDAR scan process, is now compiled by  [torch.compile](https://pytorch.org/get-started/pytorch-2.0/#pytorch-2x-faster-more-pythonic-and-as-dynamic-as-ever), which brings about 2.X speeding up. The simulation speed comparison is given as follows.

<div align="center">
<img width="65%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/FPS.png">
</div>

A more detailed comparison w.r.t. simulation speed and hardware occupation is given below.

<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/TableCompare.png">
</div>

## Features

- **[Vectorizable](https://www.gymlibrary.dev/content/vectorising/)** (Enable super fast data collection)
- **Lightweight** (Consume only 140~300 mb GPU memories for vectorized environments)
- **Standard Gym API with [Pytorch](https://pytorch.org/) data flow**
- **GPU/CPU are both acceptable** (If you use Pytorch to build your RL model, you can run your RL model and Sparrow both on GPU. Then you don't need to transfer the data from CPU to GPU anymore.)
- **Easy to use** (20kb pure Python files. Just import, never worry about installation)
- **Ubuntu/Windows/MacOS** are all supported
- **Accept image as map** (Draw your own environments easily and rapidly)
- **Detailed comments on source code**


## Installation

The dependencies for Sparrow-V1.0 are:

```bash
torch >= 2.0.1
pygame >= 2.4.0
numpy >= 1.24.3
```

You can install **torch** by following the guidance from its [official website](https://pytorch.org/get-started/locally/). We strongly suggest you install the **CUDA 11.7** (or higher) version, though CPU version or lower CUDA version is also supported.

Then you can install **pygame**, **numpy** via:

```bash
pip3 install pygame==2.4.0 numpy==1.24.3
```

Additionally, we recommended `python>=3.10.0`. Although other versions might also work. 



## Quick Start

After installation, you can play with Sparrow-V1.0 with your keyboard (UP/DOWN/LEFT/RIGHT button) to test if you have installed it successfully:

```bash
python play_with_keyboard.py
```



## Train a DDQN model with Sparrow

The Sparrow is a mobile robot simulator mainly designed for Deep Reinforcement Learning. In this section, we have prepared a simple Python script to show you how to train a [DDQN](https://ojs.aaai.org/index.php/AAAI/article/download/10295/10154) model with Sparrow-V1.0. By the way, other clean and robust Pytorch implementations of popular DRL algorithms can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).

### Start training:

To train a DDQN model with Sparrow-V1.0, you can run:

```bash
python train_DDQN_vector.py
```

By default, the above script will run on your GPU (although CPU is also supported, running Sparrow-V1.0 with GPU can be remarkably faster ). Additionally,  the script will train with the maps in `~/SparrowV1/same_maps`  with 16 vectorized environments.



### Visualize the training curve:

<img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/ep_r.svg" align="right" width="35%"/>

The above script has been incorporated with **tensorboard** to visualize the training curve, as shown on the right. To enable it, you can just set the `write` to True, e.g.

```bash
python train_DDQN_vector.py --write True
```

The training curve will be saved in the `runs` folder, for more details about how to install and use tensorboard, please click [here](https://pytorch.org/docs/stable/tensorboard.html). 



### Play with trained model:

During training, the DDQN model will be saved in the `model` folder automatically (e.g. _model/10k.pth_).  After training, you can play with it via:

```bash
python train_DDQN_vector.py --render True --Loadmodel True --ModelIdex 10 # 10 means use '10k.pth'
```

## Dive into Sparrow

### Create your first env:

Before instantiating Sparrow-V1.0, it is necessary to specify the parameters so that you can customize your own env:

```python
from SparrowV1_0.core import Sparrow, str2bool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--map_address', type=str, default='train_maps', help='same_maps / train_maps / test_maps')
parser.add_argument('--device', type=str, default='cuda:0', help='running device of sparrow')
parser.add_argument('--ld_num', type=int, default=27, help='number of lidar streams in each world')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of the world that be rendered')
parser.add_argument('--render_mode', type=str, default='human', help='human / rgb_array / None')
parser.add_argument('--render_speed', type=str, default='fast', help='real / fast / slow')
parser.add_argument('--max_ep_steps', type=int, default=1000, help='maximum episodic steps')
parser.add_argument('--AWARD', type=int, default=80, help='reward of reaching target area')
parser.add_argument('--PUNISH', type=float, default=-10, help='reward when collision happens')
parser.add_argument('--normalization', type=str2bool, default=False, help='whether normalize the observations')
parser.add_argument('--flip', type=str2bool, default=False, help='whether expand training maps with flipped maps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to torch.compile to boost simulation speed')
params = parser.parse_args()
```
Note that the number of vectorized environments is equal to the number of maps currently used. For example, here we set ```--map_address``` as **_train_maps_**. Because there are 16 maps in **_SparrowV1/train_maps_**, the default environmental copies are 16. You can create more environmental copies by putting more maps into the folder.


Afterward, you can create the Sparrow-V1.0 environment via:

```python
envs = Sparrow(params)
```

The above command will instantiate a Sparrow-V1.0 environment with standard Gym API, and you can interact with it via:

```python
import torch
device = torch.device(params.device)

s, info = envs.reset()
while True:
    a = torch.randint(0,5,(envs.N,),device=device) # 5 is the action dimension; envs.N is the number of vectorized envs
    s_next, r, terminated, truncated, info = envs.step(a)
```

Note that Sparrow-V1.0 runs in a vectorized manner, thus the dimension of **s, a, r, terminated, truncated** are **(N,32), (N,), (N,), (N,), (N,)** respectively, where **N** is the number of vectorized environments (Here, N=16). In addition, Sparrow-V1.0 has its own AutoReset mechanism. Users only need to reset the envs once.



### Coordinate Frames:

There are three coordinate frames with different orientations in Sparrow-V0, which is quite annoying. In Sparrow-V1.0, we unify the **World Coordinate Frame** and **Grid Coordinate Frame** as illustrated left and discard the Relative Coordinate Frame. In doing so, the Sparrow-V1.0 output the raw state of the robot so that users can define their own normalization method. 


<div align="center">
<img width="35%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/coordinate_frames.svg">
<img width="31%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/state.svg">
</div>

### Basic Robot Information:

The LiDAR perception range is 100cm×270°, with an accuracy of 3 cm. The radius of the robot is 9 cm, and its collision threshold is 14 cm. 


<img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/basic_robot_info.svg" align="right" width="25%"/>

The maximum linear and angular velocity of the robot is 18 cm/s and 1 rad/s, respectively. The control frequency of the robot is 10Hz. And we use a simple but useful model to describe the kinematics of the robot

$$[V^{i+1}\_{linear},\ V^{i+1}\_{angular}] = K·[V^{i}\_{linear},\ V^{i}\_{angular}]+(1-K)·[V^{target}\_{linear},\ V^{target}\_{angular}]$$

$$[dx^{i+1},dy^{i+1},\theta^{i+1}] = [dx^{i},dy^{i},\theta^{i}] + [V^{i+1}\_{linear},V^{i+1}\_{linear},\ V^{i+1}\_{angular}]·\Delta t · [\cos(\theta^{i}), -\sin(\theta^{i}), 1]$$

Here, **K** is a hyperparameter between (0,1), describing the combined effect of inertia, friction and the underlying velocity control algorithm, default: 0.6. The parameters mentioned in this section can be found in the *Robot initialization* and *Lidar initialization* part of `SparrowV1/sparrow.py` and customized according to your own scenario.


### RL representation:

The basic task in Sparrow is about driving the robot from the start point to the end point as fast as possible, without colliding with obstacles. To this end, in the following sub-sections, we will define several basic components of the Markov Decision Process.

#### State:

The state of the robot is a vector of length 32, containing **position** (*state[0:2] = [dx,dy]*), **orientation** (state[2]=θ), **velocity** (*state[3:5]=[v_linear, v_angular]*), **LiDAR** (*state[5:32] = scanning result*). Note that if the `--normalization` were set to `False` when instantiating the env, the env would output the raw state in **World Coordinate Frame**. Otherwise, the env outputs  a normalized state. For more details, please check the `_Normalize()` function in `SparrowV1/sparrow.py`.


#### Action:

There are 6 discrete actions in Sparrow, controlling the target velocity of the robot:

- **Turn Left:** [ 0.36 cm/s, 1 rad/s ]
- **Turn Left + Move forward:** [ 18 cm/s, 1 rad/s ]
- **Move forward:** [ 18 cm/s, 0 rad/s ]
- **Turn Right + Move forward:** [ 18 cm/s, -1 rad/s ]
- **Turn Right:** [ 0.36 cm/s, -1 rad/s ]
- **Stop:** [ 0 cm/s, 0 rad/s ]

We strongly suggest not using the **Stop** action when training an RL model, because it may result in the robot standing still and generating low-quality data. You might have also noted that when the robot is turning left or right, we also give it a small linear velocity. We do this to help the robot escape from the deadlock.

#### Reward:

In Sparrow-V1.0, we only provide the naive reward function.
R=80, when arrive;
R=-10, when collide;
R=0, otherwise

#### Termination:

The episode would be terminated only when the robot collides with the obstacles or reaches the target area.

#### Truncation:

The episode would be truncated only when the episode steps exceed `params.max_ep_steps`. 



### Random initialization:

At the beginning of every episode, the robot will be randomly initialized in the lower right corner of the map with different orientations to avoid overfitting.



### Simulation Speed:

If `render_mode=None` or `render_mode="rgb_array"`, Sparrow would run at its maximum simulation speed (depending on the hardware). However, if `render_mode="human"`, there would be three options regarding the simulation speed:

- `render_speed == 'fast'`: render the Sparrow in a pygame window with maximum FPS
- `render_speed == 'slow'`: render the Sparrow in a pygame window with 5 FPS. Might be useful when debugging.
- `render_speed == 'real'`: render the Sparrow in a pygame window with **1/ctrl_interval** FPS, in accordance with the real world speed.



### Customize your own maps:

Sparrow takes `.png` images as its maps, e.g. the `map0.png`~`map15.png` in `SparrowV1/train_maps/`. Therefore, you can draw your own maps with any image process software easily and conveniently, as long as it satisfies the following requirements:

- saved in `.png` format
- resolution (namely the map size) equals 366×366
- obstacles are in black (0,0,0) and free space is in white (255,255,255)
- adding a fence to surround the map so that the robot cannot run out of the map



### Number of vectorized environments:

The number of vectorized environments of Sparrow-V1.0 is exactly equal to the number of maps in `params.map_address`. You can easily create more vectorized environments by just copying more `map.png` files.



### AutoReset:

The environment copies inside the vectorized environment may be done (terminated or truncated) in different timesteps. Consequently, it is inefficient or even improper to call the *env.reset()* function to reset all copies whenever one copy is done, necessitating the design of **AutoReset** mechanism, which is illustrated below:

<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/AutoReset.svg">
</div>

**Note:**

  a) the AutoReset mechanism of Sparrow-V1.0 is different from Sparrow-V0
  
  b) different environment copies are reset independently

  c) the interaction in ```train_DDQN_vector.py``` runs in the following way:
```python
A1 = model.select_action(S1)
S2, R2, dw2, tr2, info2 = envs.step(A1)
buffer.add(S1, A1, R2, dw2, ct1)
done2 = dw2+tr2
ct2 = ~(done2)
```

## Important Differences from Sparrow-V0

Some features from Sparrow-V0 are modified in ```Sparrow-V1.0```. They are:

+ Sensor Noise: Not supported currently
+ [Domain Randomization](https://arxiv.org/pdf/1703.06907.pdf%60): Not supported currently
+ [Numpy](https://numpy.org/) data flow: Discarded
+ Map0.png with randomly generated obstacles: Discarded
+ Random initialization: ```train_maps_startpoints``` is Discarded
+ Reward Function: Replaced with sparse reward function
+ Coordinate system: Modified
+ AutoReset mechanism: Modified

## Citing the Project

To cite this repository in publications:

```bibtex
@article{Color2023JinghaoXin,
  title={Train a Real-world Local Path Planner in One Hour via Partially Decoupled Reinforcement Learning and Vectorized Diversity},
  author={Jinghao Xin, Jinwoo Kim, Zhi Li, and Ning Li},
  journal={arXiv preprint arXiv:2305.04180},
  url={https://doi.org/10.48550/arXiv.2305.04180},
  year={2023}
}
```



## Writing in the end

The name "Sparrow" actually comes from an old saying *“The sparrow may be small but it has all the vital organs.”* Hope you enjoy using Sparrow! 

Additionally, we have made detailed comments on the source code (`SparrowV1/sparrow.py`) so that you can modify Sparrow to fit your own problem. But only for non-commercial purposes, and all rights are reserved by [Jinghao Xin](https://github.com/XinJingHao).





