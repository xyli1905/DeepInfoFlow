import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class TanhX(nn.Module):
    def __init__(self, Vmax, Vmin, slope, dispX):
        super(TanhX, self).__init__()
        '''
        params:
        Vmax: max value of Output can be None
        Vmin: min value of Output can be None
        slope: slope in linear area
        dispX: displacement in X axis
        '''

        if Vmax == None and Vmin == None:
            self.method = 0
            displacementX = 0.0
            displacementY = 0.0
            scaleX = 1.0
            scaleY = 1.0
        elif Vmax == None and Vmin != None:
            self.method = 1
            displacementX = dispX
            displacementY = Vmin
            scaleY = float(slope)
            scaleX = 1.0
        elif Vmax != None and Vmin == None:
            self.method = 2
            displacementX = dispX
            displacementY = Vmax
            scaleY = float(slope)
            scaleX = 1.0
        else:# both Vmin and Vmax has value
            self.method = 3
            displacementX = dispX
            displacementY = float(0.5 * (Vmax + Vmin))
            scaleY = float(0.5 * (Vmax - Vmin))
            scaleX = float(scaleY / slope)


        displacementX = torch.tensor(displacementX, dtype = torch.float)
        displacementY = torch.tensor(displacementY, dtype = torch.float)
        scaleX = torch.tensor(scaleX, dtype = torch.float)
        scaleY = torch.tensor(scaleY, dtype = torch.float)

        self.displacementX = torch.nn.Parameter(displacementX)
        self.displacementY = torch.nn.Parameter(displacementY)
        self.scaleX = torch.nn.Parameter(scaleX)
        self.scaleY = torch.nn.Parameter(scaleY)

        self.displacementX.requires_grad = False
        self.displacementY.requires_grad = False
        self.scaleX.requires_grad = False
        self.scaleY.requires_grad = False

    def forward(self, x):
        if self.method == 0:
            return NoPoint(x)
        elif self.method == 1:
            return self.softPlus(x)
        elif self.method == 2:
            return self.negativeSoftPlus(x)
        elif self.method == 3:
            return self.Tanh(x)
        else:
            pass

    def NoPoint(self, x):
        return x

    def softPlus(self, x):
        return self.scaleY * torch.log(torch.tensor(1., dtype = torch.float) + torch.exp((1/self.scaleX) * (x - self.displacementX))) + self.displacementY

    def negativeSoftPlus(self, x):
        return -self.scaleY * torch.log(torch.tensor(1., dtype = torch.float) + torch.exp((1/self.scaleX) * (-x + self.displacementX))) + self.displacementY

    def Tanh(self, x):
        return self.scaleY * torch.tanh((1/self.scaleX) * (x - self.displacementX)) + self.displacementY

if __name__ == "__main__":
    my_net = nn.Sequential(TanhX(Vmax = 3, Vmin=-4, slope=1, dispX = 0))
    x = torch.arange(-5, 5, 0.1)
    y = my_net(Variable(x))
    import matplotlib.pyplot as plt
    print(y)
    plt.plot(x.numpy() , y.numpy())
    plt.show()