import torch
import torch.nn as nn
import torch.optim as optim

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()

        self.Dense1 = nn.Linear(12, 12)
        self.Dense2 = nn.Linear(12, 10)
        self.Dense3 = nn.Linear(10, 7)
        self.Dense4 = nn.Linear(7, 5)
        self.Dense5 = nn.Linear(5, 4)
        self.Dense6 = nn.Linear(4, 3)
        self.Dense7 = nn.Linear(3, 2)
        self.Dense8 = nn.Linear(2, 2)

    def forward(self,x):
        x = self.Dense1(x)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        x = self.Dense6(x)
        x = self.Dense7(x)
        x = self.Dense8(x)

        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)



if __name__ == '__main__':
    model = model()
    model.apply(weights_init)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    #example forward
    dummy_x = torch.randn(5, 12) #feature
    dummy_y = torch.randint(0,1,(5,2)) #label
    result = model.forward(dummy_x) #inference
    # print(result)

    criterion = nn.CrossEntropyLoss()# loss
    #example backprop
    loss = criterion(result, torch.max(dummy_y, 1)[1]) #calculate loss


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)




