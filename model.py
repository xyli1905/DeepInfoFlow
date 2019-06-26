import torch
import torch.nn as nn
import torch.optim as optim
import copy
from ActivX import ReLUX, TanhX


class Model(nn.Module):
    def __init__(self, activation , dims, train = True):
        super(Model,self).__init__()
        self.layer_dims = dims
        self.D = nn.ModuleList([])
        self.A = nn.ModuleList([])
        self._train = train
        self.construct_model_by_name(activation)


    def construct_model_by_name(self, name):
        depth = len(self.layer_dims) - 1
        numOfActiv = depth - 1

        activ_dict = {'tanh':nn.Tanh(), 'relu': nn.ReLU(), 'relu6': nn.ReLU6(), 
                      'elu': nn.ELU(), 'prelu': nn.PReLU(), 'leakyRelu': nn.LeakyReLU(), 
                      'sigmoid': nn.Sigmoid(), 'softplus': nn.Softplus()}

        ActivX_dict = {'relux': ReLUX, 'tanhx': TanhX}

        if name in activ_dict.keys():
  
            print("\rusing buildin activation")

            for i in range(depth):
                if numOfActiv > 0:
                    numOfActiv -= 1
                    self.A.append( activ_dict[name] )
                self.D.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))

        elif name in ActivX_dict.keys():

            print("\rusing activationX")

            for i in range(depth):
                if numOfActiv > 0:
                    numOfActiv -= 1
                    self.A.append( ActivX_dict[name](Vmax = 1, Vmin = -1, slope = 1, dispX = 0) )
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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setup
    print(device)
    model = Model(activation = "relux", dims = [12,6,2])
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




