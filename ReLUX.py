import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class ReLUX(nn.Module):
    def __init__(self, minVal = None, maxVal = 1.0, slope = 1.0):
        super(ReLUX, self).__init__()
        minVal = minVal if minVal != None else -float("inf")
        maxVal = maxVal if maxVal != None else float("inf")
        self.minVal = torch.tensor(minVal, dtype = torch.float)
        self.maxVal = torch.tensor(maxVal, dtype = torch.float)
        self.slope = torch.tensor(slope, dtype = torch.float)

    def forward(self, x):
        x = torch.min(torch.max(x, self.minVal), self.maxVal) * self.slope
        return x


if __name__ == "__main__":
    my_net = nn.Sequential(ReLUX())
    y = my_net(Variable(torch.tensor(-1.5)))
    print(y)