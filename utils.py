#import keras
#import keras.backend as K
import numpy as np
import scipy.io as sio
from pathlib2 import Path
from collections import namedtuple
import torch
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.datasets as data
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, ID, train=True):
        self.nb_classes = 2
        data_file = Path('datasets/IB_data_'+str(ID)+'.npz')
        if data_file.is_file():
            data = np.load('datasets/IB_data_'+str(ID)+'.npz')
        else:
            create_IB_data(ID)
            data = np.load('datasets/IB_data_'+str(ID)+'.npz')
        
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


def get_mnist():
    """ nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32') / 255.
    X_test  = np.reshape(X_test , [X_test.shape[0] , -1]).astype('float32') / 255.
    #X_train = X_train * 2.0 - 1.0
    #X_test  = X_test  * 2.0 - 1.0

    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes).astype('float32')
    Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes).astype('float32')

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)

    del X_train, X_test, Y_train, Y_test, y_train, y_test """
    training_set = data.MNIST(root='./datasets', train=True, transform = transforms.ToTensor(), target_transform = None, download=True)
    testing_set = data.MNIST(root='./datasets', train=False, transform = transforms.ToTensor(), target_transform = None, download = True)

 
    return training_set, testing_set

def create_IB_data(idx):
    data_sets_org = load_data()
    data_sets = data_shuffle(data_sets_org, [80], shuffle_data=True)
    X_train, y_train, X_test, y_test = data_sets.train.data, data_sets.train.labels[:,0], data_sets.test.data, data_sets.test.labels[:,0]
    np.savez_compressed('datasets/IB_data_'+str(idx), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

def construct_full_dataset(trn, tst):
    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    X = np.concatenate((trn.X,tst.X))
    y = np.concatenate((trn.y,tst.y))
    Y = np.concatenate((trn.Y,tst.Y))
    return Dataset(X, Y, y, trn.nb_classes)
 
def load_data():
    """Load the data
    name - the name of the dataset
    return object with data and labels"""
    print ('Loading Data...')
    C = type('type_C', (object,), {})
    data_sets = C()
    d = sio.loadmat('datasets/var_u.mat')
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    data_sets = C()
    data_sets.data = F
    data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    return data_sets

def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
    start_test_index = stop_train_index
    if percent_of_train[0] > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:, :]
    return data_sets

if __name__ == "__main__":
    # get_IB_data('2017_12_21_16_51_3_275766')
    # test = CustomDataset('2017_12_21_16_51_3_275766')
    train, test = get_mnist()
    dataset_loader = torch.utils.data.DataLoader(test, batch_size=4, shuffle=True, num_workers=4)
    l = 0
    for i, (x, y) in enumerate(dataset_loader):
        print(x)
        print(y)
        l += 1
        if l == 5:
            break

            
    # import requests
    # import pandas
    # import json

    # headers={ 
    # "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36", 
    # } 
    # url = 'https://stats.nba.com/stats/commonallplayers'
    # params = {"LeagueId": "00", "Season":'2016-17', 'IsOnlyCurrentSeaSon':'0'} 
    # response = requests.get(url=url, params=params, headers=headers).text 
    # data = json.loads(response)
    # print(data)
