import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class DataLoadSplit():

    def __init__(self, encoded_labels, features):
        self.encoded_labels = encoded_labels
        self.features = features
        self.split_frac = 0.8
        self.batch_size = 50
    
    def load_split_data(self):

        ## split data into training, validation, and test data (features and labels, x and y)
        nb_row = self.features.shape[0]
        train_x = self.features[:int(nb_row*self.split_frac), :]
        test_x = self.features[int(nb_row*self.split_frac):int((nb_row*self.split_frac + ((nb_row-nb_row*self.split_frac)/2))), :]
        val_x = self.features[int((nb_row*self.split_frac + ((nb_row-nb_row*self.split_frac)/2))):, :]

        train_y = np.asarray(self.encoded_labels[:int(nb_row*self.split_frac)])
        test_y = np.asarray(self.encoded_labels[int(nb_row*self.split_frac):int((nb_row*self.split_frac + ((nb_row-nb_row*self.split_frac)/2)))])
        val_y = np.asarray(self.encoded_labels[int((nb_row*self.split_frac + ((nb_row-nb_row*self.split_frac)/2))):])

        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        # make sure to SHUFFLE your data
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size)

        return train_loader, test_loader, valid_loader



def train():
    train_on_gpu=torch.cuda.is_available()

    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')