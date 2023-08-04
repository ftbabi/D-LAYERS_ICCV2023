# D-LAYERS Dataset
\[[Paper](https://arxiv.org/abs/2305.10418)\] | \[[Project](https://mmlab-ntu.github.io/project/layersnet/index.html)\] | \[[Code](https://github.com/ftbabi/LayersNet_ICCV2023.git)\]

![](imgs/demo_rotate.gif)
![](imgs/demo_wind.gif)

This is the repository of Dataset for "Towards Multi-Layered 3D Garments Animation, ICCV 2023". Please refer to our [project page](https://mmlab-ntu.github.io/project/layersnet/index.html) for more details.

**Authors**: Yidi Shao, [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/),  and [Bo Dai](http://daibo.info/).

**Acknowedgement**: This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOE-T2EP20221-0011) and Shanghai AI Laboratory. 

## Access of Dataset
Please download the dataset [here]().

## Demo
1. Download the data samples from [here](https://drive.google.com/file/d/1X7cwuyy_6HPh05cq0v8U5i43012UasC7/view?usp=sharing) and unzip the file into this repo. You should see a directory named `data` with `demo` and `smpl` inside.
2. Download SMPL models in this [page](https://smpl.is.tue.mpg.de/). Name the file as `model_f.pkl` and `model_m.pkl`, then save into `data/smpl`.
3. To view the examples in `demo.ipynb`, please create the following environment.
```
conda create -n LAYERS python=3.8
conda activate LAYERS

# Dependent packages
pip3 install numpy==1.23.1 scipy chumpy plotly ipykernel nbformat
```

## Citations
```
@inproceedings{shao2023layersnet,
  author = {Shao, Yidi and Loy, Chen Change and Dai, Bo},
  title = {Towards Multi-Layered {3D} Garments Animation},
  booktitle = {ICCV},
  year = {2023}
}
```
