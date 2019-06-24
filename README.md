# Deep Information flow

## Introduction
Repo for our project Deep Information Flow<br/>
We follow the idea from [ibsgd](https://github.com/artemyk/ibsgd) project. we reproduce their result by [PyTorch](https://pytorch.org/) which is faster than original Keras code. Despite the original mutual information (MI) measurement we also propose our own EVKL method in this code for comparison (it is still ongoing and the code is not stable right now).
**This project is still ongoing. Detailed description will come soon**

## Prerequisite

```
numpy
matplotlib
pytorch
torchvision
pathlib2
```

## Usage
1. The first step is to train a fully-connected deep neural network for analysis and calculate MI later.
```
python IBNet.py
```
The settings and configurations such as *batch size*, *learning rate* and *max epochs* are stored in [base_options.py](./base_options.py)
