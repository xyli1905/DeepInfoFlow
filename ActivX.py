import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class ActivationX(nn.Module):
    def __init__(self, Vmax, Vmin, slope, dispX):
        super(ActivationX, self).__init__()
        '''
        ActivationX is the base class for ReLUX and TanhX.  

        We generally summarise an activation by the following properties:
        1. Limits: Vmax and Vmin. 
            (1) if both apply, then it's a double-saturated activation;
            (2) if either Vmax or Vmin applies, then it's a single-saturated activation;
            (3) if none applies, then we assume it is the straight line: y = x;
        2. slope (see code for detail)
            (1) for ReLUX, just the slope of the oblique part;
            (2) for TanhX, we either use the slope at +-infty or at 'center';
        3. Position: dispX (displacement along x-axis)
            (1) we define reference activations for each case, and specify the position
                througth the relative displacement;
            (2) it turns out, given Vmax and Vmin, only dispX can specify the relative
                position to the reference;

        The number of reference activations is arbitrary in principle, presently we 
        use three, corresponding to left-saturated, right-saturated and double-saturated
        cases.
        '''
        self._name = 'base activation X'
        if Vmax != None and Vmin != None:
            assert Vmax > Vmin, "Vmax should be larger than Vmin"

        activType, deltaX, deltaY, scaleX, scaleY = self.get_param(Vmax, Vmin, slope, dispX)

        if activType == 0:
            self.base_activ = self.activ_double_sat
        elif activType == 1:
            self.base_activ = self.activ_left_sat
        elif activType == 2:
            self.base_activ = self.activ_right_sat
        elif activType == 3:
            self.base_activ = self.a_line

        deltaX = torch.tensor(deltaX, dtype = torch.float)
        deltaY = torch.tensor(deltaY, dtype = torch.float)
        scaleX = torch.tensor(scaleX, dtype = torch.float)
        scaleY = torch.tensor(scaleY, dtype = torch.float)

        self.deltaX = torch.nn.Parameter(deltaX)
        self.deltaY = torch.nn.Parameter(deltaY)
        self.scaleX = torch.nn.Parameter(scaleX)
        self.scaleY = torch.nn.Parameter(scaleY)

        self.deltaX.requires_grad = False
        self.deltaY.requires_grad = False
        self.scaleX.requires_grad = False
        self.scaleY.requires_grad = False


    def get_param(self, Vmax, Vmin, slope, dispX):
        if Vmax != None and Vmin != None:
            activType = 0
            deltaX = dispX
            deltaY = float(0.5 * (Vmax + Vmin))
            scaleY = float(0.5 * (Vmax - Vmin))
            scaleX = float(scaleY / slope)
        elif Vmax == None and Vmin != None:
            activType = 1
            deltaX = dispX
            deltaY = Vmin
            scaleY = float(slope)
            scaleX = 1.0
        elif Vmax != None and Vmin == None:
            activType = 2
            deltaX = dispX
            deltaY = Vmax
            scaleY = float(slope)
            scaleX = 1.0
        else:# None of Vmin and Vmax has value
            activType = 3
            deltaX = 0.0
            deltaY = 0.0
            scaleX = 1.0
            scaleY = 1.0

        return activType, deltaX, deltaY, scaleX, scaleY


    def activ_double_sat(self, input):
        raise NotImplementedError('to be overrided')

    def activ_left_sat(self, input):
        raise NotImplementedError('to be overrided')

    def activ_right_sat(self, input):
        raise NotImplementedError('to be overrided')

    def a_line(self, input):
        return input


    def forward(self, input):
        normedX = (input - self.deltaX) / self.scaleX #first scale then displace
        return self.scaleY * self.base_activ( normedX ) + self.deltaY



class ReLUX(ActivationX):
    def __init__(self, Vmax = None, Vmin = None, slope = 1.0, dispX = 0.0):
        super(ReLUX, self).__init__(Vmax, Vmin, slope, dispX)
        self._name = 'ReLU X'
        self.ONE = torch.tensor(1.0, dtype = torch.float)
        self.ZERO = torch.tensor(0.0, dtype = torch.float)

    def activ_double_sat(self, input):
        return torch.min(torch.max(input, -self.ONE), self.ONE)

    def activ_left_sat(self, input):
        return torch.max(input, self.ZERO)

    def activ_right_sat(self, input):
        return torch.min(input, self.ZERO)



class TanhX(ActivationX):
    def __init__(self, Vmax = None, Vmin = None, slope = 1.0, dispX = 0.0):
        super(TanhX, self).__init__(Vmax, Vmin, slope, dispX)
        self._name = 'Tanh X'

    def activ_double_sat(self, input):
        return torch.tanh(input)

    def activ_left_sat(self, input):
        softplus = nn.Softplus()
        return softplus(input)

    def activ_right_sat(self, input):
        softplus = nn.Softplus()
        return -1.0*softplus(-1.0*input)




if __name__ == "__main__":
    my_net = nn.Sequential(ReLUX(Vmax = 5, Vmin=1, slope=1, dispX = 2))
    x = torch.arange(-10, 10, 0.1)
    y = my_net(Variable(x))
    import matplotlib.pyplot as plt
    # print(y)
    plt.plot(x.numpy() , y.numpy())
    plt.show()