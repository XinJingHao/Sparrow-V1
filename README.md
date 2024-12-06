
<div align="center">
  <a ><img width="300px" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/LOGO%20sparrow.jpg"></a>
</div>

## Sparrow-V1.2: A Reinforcement Learning Friendly Simulator for Mobile Robot

<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/render.gif">
</div>

![Python](https://img.shields.io/badge/Python-blue)
![DRL](https://img.shields.io/badge/DRL-blueviolet)
![Mobile Robot](https://img.shields.io/badge/MobileRobot-ff69b4)


## Main Differences from [Sparrow-V1.1](https://github.com/XinJingHao/Sparrow-V1/tree/Sparrow-V1.1):
+ ```state``` is changed to [ $$D2T$$, $$\alpha$$, $$V_{linear}$$, $$V_{angular}$$, LiDAR ]
+ state normalization is mandatory
+ $V^{max}_{linear}$ from 18 cm/s to 50 cm/s
+ $V^{max}_{angular}$ from 1 rad/s to 2 cm/s



## Citing the Project

To cite this repository in publications:

```bibtex
@article{Color2025XJH,
title = {Train a real-world local path planner in one hour via partially decoupled reinforcement learning and vectorized diversity},
journal = {Engineering Applications of Artificial Intelligence},
volume = {141},
pages = {109726},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2024.109726},
}
```



## Writing in the end

The name "Sparrow" actually comes from an old saying *“The sparrow may be small but it has all the vital organs.”* Hope you enjoy using Sparrow! 

Additionally, we have made detailed comments on the source code (`SparrowV1_2/core.py`) so that you can modify Sparrow to fit your own problem. But only for non-commercial purposes, and all rights are reserved by [Jinghao Xin](https://github.com/XinJingHao).





