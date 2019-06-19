import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class ReLUX(nn.Module):
    def __init__(self,leftPoint = None, rightPoint = None):
        super(ReLUX, self).__init__()

        '''
        params:
        leftPoint: [x, y] coordinates of left bent point
        rightPoint: [x, y] coordinates of right bent point
        '''

        minVal, maxVal, slope, bias = self.getParameter(leftPoint = leftPoint, rightPoint = rightPoint)

        tensorMinVal = torch.tensor(minVal, dtype = torch.float) if minVal != None else torch.tensor(-float("inf"), dtype = torch.float)
        tensorMaxVal = torch.tensor(maxVal, dtype = torch.float) if maxVal != None else torch.tensor(float("inf"), dtype = torch.float)
        tensorSlope = torch.tensor(slope, dtype = torch.float)
        tensorBias = torch.tensor(bias, dtype = torch.float)

        self.minVal = torch.nn.Parameter(tensorMinVal)
        self.maxVal = torch.nn.Parameter(tensorMaxVal)
        self.slope = torch.nn.Parameter(tensorSlope)
        self.shift = torch.nn.Parameter(tensorBias) # do not change the name to self.bias this is the same as bias

        self.minVal.requires_grad = False
        self.maxVal.requires_grad = False
        self.slope.requires_grad = False
        self.shift.requires_grad = False

    def forward(self, x):
        x = torch.min(torch.max(x * self.slope, self.minVal), self.maxVal) + self.shift
        return x

    def getParameter(self, leftPoint, rightPoint):
        if leftPoint == None and rightPoint == None:
            return None, None, 1.0, 0.0
        elif leftPoint == None and rightPoint != None:
            return None, rightPoint[0], 1, rightPoint[1]
        elif leftPoint != None and rightPoint == None:
            return leftPoint[0], None, 1, leftPoint[1]
        else:
            slope = float((rightPoint[1] - leftPoint[1]) / (rightPoint[0] - leftPoint[0])) #dY/dX
            bias = float(min(leftPoint[1], rightPoint[1]))
            minVal = float(leftPoint[1])
            maxVal = float(rightPoint[1])
            return minVal, maxVal, slope, bias


if __name__ == "__main__":
    my_net = nn.Sequential(ReLUX(leftPoint = None, rightPoint = [0, 0]))
    y = my_net(Variable(torch.tensor(-2.5)))
    print(y)