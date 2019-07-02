import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as data
import torchvision.transforms as transforms
import os
import numpy as np
import struct



class NumericalDataset(Dataset):
    def __init__(self, path, train=True):
        if os.path.isfile(path):
            data = np.load(path)
        else:
            raise RuntimeError("no such file")
        
        if train:
            (X, Y) = (data['X_train'], data['y_train'])
        else:
            (X, Y) = (data['X_test'], data['y_test'])
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
        assert len(self.X) == len(self.Y)
        

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.Y[index]
        return (feature, label)

    def __len__(self):
        return len(self.Y)

class DataProvider:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs.get('dataset_name', 'IBNet')
        self.batch_size = kwargs.get('batch_size', 1)
        self.num_workers = kwargs.get('num_workers', 0)
        self.shuffle = kwargs.get('shuffle', True)
        self._generate_data()

    def _generate_data(self):
        self.train_set = None
        self.test_set = None
        if self.dataset_name == 'IBNet':
            self.train_set = NumericalDataset('./datasets/IB_data_2017_12_21_16_51_3_275766.npz', train=True)
            self.test_set = NumericalDataset('./datasets/IB_data_2017_12_21_16_51_3_275766.npz', train=False)
        elif self.dataset_name == 'MNIST':
            self.train_set = data.MNIST(root='./datasets', train=True, transform = transforms.ToTensor(), target_transform = None, download = True)
            self.test_set = data.MNIST(root='./datasets', train=False, transform = transforms.ToTensor(), target_transform = None, download = True)

        if self.train_set == None or self.test_set == None:
            raise RuntimeError('No such dataset')

    def get_train_test_data(self):
        train_loader = DataLoader(self.train_set, batch_size = self.batch_size, shuffle = self.shuffle, num_workers = self.num_workers)
        test_loader = DataLoader(self.test_set, batch_size = self.batch_size, shuffle = self.shuffle, num_workers = self.num_workers)

        return train_loader, test_loader

    def get_full_data(self):
        full_data = torch.utils.data.ConcatDataset([self.test_set, self.train_set])
        full_data = DataLoader(full_data, batch_size = self.batch_size, shuffle = self.shuffle, num_workers = self.num_workers)
        return full_data

if __name__ == "__main__":
    test = DataProvider(dataset_name = "IBNet", batch_size = 1, num_workers = 0, shuffle = True)
    train_d= test.get_full_data()
    print(len(train_d))





