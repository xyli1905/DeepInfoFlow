import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class ReLUX(nn.Module):
    def __init__(self, minVal = 0.0, maxVal = None, slope = 1.0, bias = 0.0):
        super(ReLUX, self).__init__()

        tensorMinVal = torch.tensor(minVal, dtype = torch.float) if minVal != None else torch.tensor(-float("inf"), dtype = torch.float)
        tensorMaxVal = torch.tensor(maxVal, dtype = torch.float) if maxVal != None else torch.tensor(float("inf"), dtype = torch.float)
        tensorSlope = torch.tensor(slope, dtype = torch.float)
        tensorBias = torch.tensor(bias, dtype = torch.float)

        self.minVal = torch.nn.Parameter(tensorMinVal)
        self.maxVal = torch.nn.Parameter(tensorMaxVal)
        self.slope = torch.nn.Parameter(tensorSlope)
        self.bias = torch.nn.Parameter(tensorBias)

        self.minVal.requires_grad = False
        self.maxVal.requires_grad = False
        self.slope.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        x = torch.min(torch.max(x * self.slope, self.minVal), self.maxVal) + self.bias
        return x


if __name__ == "__main__":
    my_net = nn.Sequential(ReLUX())
    y = my_net(Variable(torch.tensor(-1.5)))
    print(y)