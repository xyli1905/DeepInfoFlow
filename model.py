import torch
import torch.nn as nn
import torch.optim as optim
import copy

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.layer_dims = [12, 12, 10, 7, 5, 4, 3, 2, 2]
        self.D = {}
        self.A = {}
        self.construct_model_by_name('relu')


    def construct_model_by_name(self, name):
        depth = len(self.layer_dims) - 1
        numOfActiv = depth - 1

        if name == 'tanh':
            for i in range(depth):
                setattr(self, 'Dense' + str(i), nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
                if numOfActiv > 0:
                    setattr(self, 'Activ' + str(i), nn.Tanh())
                    numOfActiv -= 1
                    self.A[i] = nn.Tanh()
                self.D[i] = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
        elif name == 'relu':
            for i in range(depth):
                setattr(self, 'Dense' + str(i), nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
                if numOfActiv > 0:
                    setattr(self, 'Activ' + str(i), nn.ReLU())
                    numOfActiv -= 1
                    self.A[i] = nn.ReLU()
                self.D[i] = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
 

    def forward(self, x):
        for i in range(len(self.layer_dims) - 1):
            dense = self.D[i]
            x = dense(x)
            if i < len(self.A):
                activ = self.A[i]
                x = activ(x) 

        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)



if __name__ == '__main__':
    model = model()
    print (model)
    model.apply(weights_init)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    #example forward
    dummy_x = torch.randn(5, 12) #feature
    dummy_y = torch.randint(0,1,(5,2)) #label
    result = model.forward(dummy_x) #inference
    print(result)

    criterion = nn.CrossEntropyLoss()# loss
    #example backprop
    loss = criterion(result, torch.max(dummy_y, 1)[1]) #calculate loss


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)




