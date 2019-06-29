import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# --------------------------- #
# Classes for Networks        #
# --------------------------- #
class BaseNetwork(nn.Module):
    def __init__(self, opt, train = True):
        super(BaseNetwork, self).__init__()
        self._name = 'base network'
        self._opt = opt
        self._train = train

        # dictionary of available activations
        #NOTE the use of eval in DenseNet
        self.activ_dict = {'tanh': 'nn.Tanh()', 'relu': 'nn.ReLU()', 'relu6': 'nn.ReLU6()', 
                           'elu': 'nn.ELU()', 'prelu': 'nn.PReLU()', 'leakyRelu': 'nn.LeakyReLU()', 
                           'sigmoid': 'nn.Sigmoid()', 'softplus': 'nn.Softplus()'}

        self.ActivX_dict = {'relux': 'ReLUX', 'tanhx': 'TanhX'}

        self.construct_model()

    def construct_model(self):
        raise NotImplementedError('not defined in BaseNetwork')

    def forward(self, input):
        raise NotImplementedError('not defined in BaseNetwork')

# Dense net or fully-connected net
class DenseNet(BaseNetwork):
    # def __init__(self, activation , dims, train = True):
    def __init__(self, opt, train = True):
        super(DenseNet, self).__init__(opt, train)
        self.layer_dims = self._opt.layer_dims
        self.D = nn.ModuleList([])
        self.A = nn.ModuleList([])

    def construct_model(self):
        name = self._opt.activation
        depth = len(self.layer_dims) - 1
        numOfActiv = depth - 1

        if name in self.activ_dict.keys():
  
            print("\rusing buildin activation")

            for i in range(depth):
                if numOfActiv > 0:
                    numOfActiv -= 1
                    self.A.append( eval(self.activ_dict[name]) )
                self.D.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))

        elif name in self.ActivX_dict.keys():

            print("\rusing activationX")

            for i in range(depth):
                if numOfActiv > 0:
                    numOfActiv -= 1
                    self.A.append( eval(self.ActivX_dict[name])(Vmax = self._opt.Vmax, 
                                                                Vmin = self._opt.Vmin, 
                                                                slope = self._opt.slope, 
                                                                dispX = self._opt.dispX))
                self.D.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))

        else:
            raise RuntimeError('Do not have {activation} activation function please check your options'.format(activation = name))


    def forward(self, x):
        if len(x.shape) > 2:
            print('dsfds')
            x = x.reshape(x.shape[0], -1)
        if self._train:
            for i in range(len(self.layer_dims) - 1):
                dense = self.D[i]
                x = dense(x)
                if i < len(self.A):
                    activ = self.A[i]
                    x = activ(x)
            return x
        else:
            outputs = []
            for i in range(len(self.layer_dims) - 1):
                dense = self.D[i]
                x = dense(x)
                if i < len(self.A):
                    activ = self.A[i]
                    x = activ(x)
                outputs.append(x)
            return outputs


# ---------------------------------------- #
# Classes for ActivationX: ReLUX and TanhX #
# ---------------------------------------- #
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




if __name__ == '__main__':
    # ------------------------- #
    # TEST for DenseNet
    # ------------------------- #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setup
    print(device)

    C = type('type_C', (object,), {})
    opt = C()
    opt.layer_dims = [12, 6, 2]
    opt.activation = 'tanh'

    model = DenseNet(opt)
    print (model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #example forward
    dummy_x = torch.randn(5, 12) #feature
    dummy_y = torch.randint(0,1,(5,2)) #label
    dummy_x = dummy_x.to(device)
    dummy_y = dummy_y.to(device)
    result = model.forward(dummy_x) #inference
    print(result)

    criterion = nn.CrossEntropyLoss()# loss
    #example backprop
    loss = criterion(result, torch.max(dummy_y, 1)[1]) #calculate loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)


    # ------------------------- #
    # TEST for ActivX
    # ------------------------- #
    my_net = nn.Sequential(ReLUX(Vmax = 5, Vmin=1, slope=1, dispX = 2))
    x = torch.arange(-10, 10, 0.1)
    y = my_net(Variable(x))
    import matplotlib.pyplot as plt
    # print(y)
    plt.plot(x.numpy() , y.numpy())
    plt.show()



