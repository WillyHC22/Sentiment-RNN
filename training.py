import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
#from models.model import sentimentRNN


class DataLoadSplit():

    def __init__(self, encoded_labels, features, args):
        self.encoded_labels = encoded_labels
        self.features = features
        self.split_frac = 0.8
        self.batch_size = args["batch_size"]
    
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

        loaders = {"train_loader":train_loader, "test_loader": test_loader, "valid_loader": valid_loader}
        return loaders


class TrainingRNN():
    def __init__(self, model, loaders, args):
        self.model = model
        self.train_loader = loaders["train_loader"]
        self.test_loader = loaders["test_loader"]
        self.valid_loader = loaders["valid_loader"]
        self.args = args


    def check_gpu(self):
        train_on_gpu=torch.cuda.is_available()

        if(train_on_gpu):
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')
        return train_on_gpu

    def trainRNN(self, criterion, optimizer):

        train_on_gpu = self.check_gpu()
        if train_on_gpu:
            self.model.cuda()

        self.model.train()
        counter = 0
        clip = 5
        for e in tqdm(range(self.args["epochs"])):
            h = self.model.init_hidden(self.args["batch_size"])

            for inputs, labels in tqdm(self.train_loader):
                counter += 1
                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                h = tuple([each.data for each in h])
                self.model.zero_grad()
                output, h = self.model(inputs, h)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward(retain_graph = True)
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                optimizer.step()

                if counter % self.args["print_every"] == 0:
                    val_h = self.model.init_hidden(self.args["batch_size"])
                    val_losses = []
                    self.model.eval()
                    for inputs, labels in self.valid_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        if(train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output, val_h = self.model(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                    self.model.train()
                    print("Epoch: {}/{}...".format(e+1, self.args["epochs"]),
                        "Step: {}...".format(counter),
                        "Loss: {:.6f}...".format(loss.item()),
                        "Val Loss: {:.6f}".format(np.mean(val_losses)))



