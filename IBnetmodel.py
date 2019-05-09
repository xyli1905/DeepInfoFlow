import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils
import time
import sys
import os

class IBnetModel(nn.Module):
    def __init__(self, dims, acttype = 'tanh', is_train = True):
        super(IBnetModel, self).__init__()
        self._is_train = is_train
        self.layer_dims = dims
        self._depth = len(self.layer_dims) - 1
        self.D = nn.ModuleList([])
        self._construct_model_by_name(acttype)
        self.A = nn.ModuleList([])
        # if acttype == 'tanh':
        #     self.linear1 = nn.Linear(12, 12)
        #     self.linear2 = nn.Linear(12, 10)
        #     self.linear3 = nn.Linear(10, 7)
        #     self.linear4 = nn.Linear(7, 5)
        #     self.linear5 = nn.Linear(5, 4)
        #     self.linear6 = nn.Linear(4, 3)
        #     self.linear7 = nn.Linear(3, 2)
        #     self.linear8 = nn.Linear(2, 2)
        #     self.activation = nn.Tanh()
        # elif acttype == 'relu':
        #     raise NotImplementedError('to be finished')
        # else:
        #     raise ValueError('not valid type')

    def _construct_model_by_name(self, name):
        if name == 'tanh':
            self.activation = nn.Tanh()
            for i in range(self._depth):
                self.D.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
        elif name == 'relu':
            self.activation = nn.ReLU()
            for i in range(self._depth):
                self.D.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))

    def forward(self, x):
        if self._is_train:
            for i in range(self._depth):
                x = self.D[i](x)
                if i < self._depth:
                    x = self.activation(x)
            return x
        else:
            layer_output = []
            for i in range(self._depth):
                x = self.D[i](x)
                if i < self._depth:
                    x = self.activation(x) 
                layer_output.append(x)
            return layer_output
        # layer_output = []
        # x = self.linear1(x)
        # # h = F.tanh(x)
        # h = self.activation(x)
        # layer_output.append(h)
        # x = self.linear2(h)
        # h = self.activation(x)
        # layer_output.append(h)
        # x = self.linear3(h)
        # h = self.activation(x)
        # layer_output.append(h)
        # x = self.linear4(h)
        # h = self.activation(x)
        # layer_output.append(h)
        # x = self.linear5(h)
        # h = self.activation(x)
        # layer_output.append(h)
        # x = self.linear6(h)
        # h = self.activation(x)
        # layer_output.append(h)
        # x = self.linear7(h)
        # h = self.activation(x)
        # layer_output.append(h)
        # output = self.linear8(h)
        # layer_output.append(output)
        # # x = self.linear8(h)
        # # output = F.softmax(x)
        # if self._is_train:
        #     return output
        # else:
        #     return layer_output



class Train:
    def __init__(self):
        self._name = 'train_testIBnet'
        self._fdir = "/Users/xyli1905/Projects/DeepInfoFlow/results/testIBnet"
        self._save_step = 100

        if not os.path.exists(self._fdir):
            os.mkdir(self._fdir)

        self._layer_dims = [12, 12, 10, 7, 5, 4, 3, 2, 2]
        self._acttype = 'tanh'
        self._is_train = True
        self._num_epoch = 1000
        self._batch_size = 256
        self._lr = 0.0004

        self._build_model()
        self._initialize_model()

        # set training dataset
        train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
        self._train_set = torch.utils.data.DataLoader(train_data, 
                                                      batch_size=self._batch_size, 
                                                      shuffle=True, 
                                                      num_workers=1)
        print("\nIBnet experiment:\n")

    def train(self):
        # train the model
        eta = 1.
        running_loss = 0.0
        running_acc = 0.0
        t_begin = time.time()
        for i_epoch in range(self._num_epoch):
            # set to train
            self._model.train()

            # train batch
            if ((i_epoch+1) % self._save_step == 0) or (i_epoch == 0):
                print('\n{}'.format(11*'------'))

            for i_batch , (inputs, labels) in enumerate(self._train_set):
                bsize = inputs.shape[0]
                # set to learnable
                with torch.set_grad_enabled(True):
                    #forward
                    outputs = self._model(inputs)
                    loss = self._loss(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    corrects = torch.sum(preds == labels.data).double()

                    # backprop
                    self._opt.zero_grad()
                    loss.backward()
                    self._opt.step()

                # monitor the running loss & running accuracy
                eta = eta / (1. + bsize*eta)
                running_loss = (1. - bsize*eta)*running_loss + eta*loss.detach()
                running_acc = (1. - bsize*eta)*running_acc + eta*corrects.detach()
                if ((i_epoch+1) % self._save_step == 0) or (i_epoch == 0):
                    sys.stdout.flush()
                    print('\repoch:{epoch} batch:{batch:2d} Loss:{loss:.5e} Acc:{acc:.5f}% numacc:{num:.0f}/{tnum:.0f}'\
                            .format(batch=i_batch+1,
                                    epoch=i_epoch+1, 
                                    loss=running_loss, 
                                    acc=running_acc*100., 
                                    num=corrects, 
                                    tnum=bsize)
                         )

            if ((i_epoch+1) % self._save_step == 0) or (i_epoch == 0):
                print('{}'.format(11*'------'))        
                # save model
                self._save_model(i_epoch)
                t_end = time.time()
                print('time cost for this output interval is {:.3f}(s)'.format(t_end - t_begin))
                t_begin = time.time()

    def _build_model(self):
        self._model = IBnetModel(dims = self._layer_dims, 
                                 acttype = self._acttype, 
                                 is_train = self._is_train)
        self._opt = optim.Adam(self._model.parameters(), lr = self._lr)
        # self._opt = optim.SGD(self._model.parameters(), lr = self._lr, momentum=0.9)
        self._loss = nn.CrossEntropyLoss()

    def _initialize_model(self):
        # weight initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        self._model.apply(weights_init)

    def _save_model(self, epoch):
        fname = 'model_epoch_{}.pth'.format(str(epoch+1))
        save_full_path = os.path.join(self._fdir, fname)
        torch.save({'epoch': epoch,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._opt.state_dict(),
                    }, 
                    save_full_path)
        print('model saved at {}'.format(save_full_path))



class TestDebug:
    def __init__(self):
        self._name = 'testDebug_IBnet'
        self._fdir = "/Users/xyli1905/Projects/DeepInfoFlow/results/testIBnet"

        self._layer_dims = [12, 12, 10, 7, 5, 4, 3, 2, 2]
        self._acttype = 'tanh'
        self._is_train = False
        self._batch_size = 256

        self._model = IBnetModel(dims = self._layer_dims, 
                                 acttype = self._acttype, 
                                 is_train = self._is_train)

        test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
        self._test_set  = torch.utils.data.DataLoader(test_data,
                                                      batch_size=self._batch_size, 
                                                      shuffle=True, 
                                                      num_workers=1)
        print("\nIBnet debug:\n")

    def check_val(self):
        # epoch_files = os.listdir(self._fdir)
        # for epoch_file in epoch_files:
        epoch_num = 10
        epoch_file = "model_epoch_{}.pth".format(str(epoch_num))
        fpath = os.path.join(self._fdir, epoch_file)
        ckpt = torch.load(fpath)
        self._model.load_state_dict(ckpt['model_state_dict'])
        # print(ckpt['model_state_dict'])
        epoch = ckpt['epoch']

        # set model to eval
        self._model.eval()

        layer_activity = []
        X = []
        Y = []
        for j, (inputs, labels) in enumerate(self._test_set):
            outputs = self._model(inputs)
            # print(outputs[5])
            # print(outputs[6])
            Y.append(labels)
            X.append(inputs)
            for i in range(len(outputs)):
                data = outputs[i]
                if len(layer_activity) < len(outputs):
                    layer_activity.append(data)
                else:
                    layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)
        print('-----------------------------')
        print(layer_activity[6][101])
        print('-----------------------------')
        
        # for layer in layer_activity:
        #     print('-----------------------------')
        #     print(layer[100])
        #     print('-----------------------------')
        #     # test = self.measure.entropy_estimator_kl(layer, 0.001)
        #     # print(test)




if __name__ == "__main__":
    #train IBnet
    testIBnet = Train()
    testIBnet.train()

    #test debug
    checkIBnet = TestDebug()
    checkIBnet.check_val()


# epoch:100 batch: 9 Loss:2.73744e-03 Acc:52.41546% numacc:158/256
# epoch:100 batch:10 Loss:2.73738e-03 Acc:52.41633% numacc:137/256
# epoch:100 batch:11 Loss:2.73734e-03 Acc:52.41444% numacc:128/256
# epoch:100 batch:12 Loss:2.73728e-03 Acc:52.41927% numacc:150/256
# epoch:100 batch:13 Loss:2.73763e-03 Acc:52.42187% numacc:116/205
# -----------------------------------------------------------------
# model saved at /Users/xyli1905/Projects/DeepInfoFlow/results/testIBnet/model_epoch_100.pth

# IBnet debug:

# -----------------------------
# tensor([-0.1090,  0.1376], grad_fn=<SelectBackward>)