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
pathlib2 (to be deprecated)
moviepy
imageio=2.4.1
requests
```

## Usage
1. The first step is to train a fully-connected deep neural network for analysis and calculate MI later.
```
python IBNet.py
```
The settings and configurations such as *batch size*, *learning rate*... are stored in [base_options.py](./base_options.py). To be noticed, the *layer_dims* parameter represents the architecture of the network. It takes a list as input. For example *[12, 6, 2]* means a network with 12 dimension inputs, 6 dimension hidden layer and 2 dimension outputs. The outputs of the trained network would be stored under results folder with name and time stamp.

2. For measuring MI.
```
python ComputeMI.py
```
This python script will automatically find the newest trained neural networks under results folder and measure the MI. The result will be stored in the plot folder under the neural networks folder.

## GUI
We also provide a simple QT GUI for easy training and measuring. It is under constructing right now.

