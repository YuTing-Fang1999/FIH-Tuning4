
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import os

from .model import My_Model

class My_Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y):
        self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # y = np.tanh(self.y[idx])
        return x, y

    def __len__(self):
        return len(self.y)


class ML(QWidget):
    # logger
    log_info_signal = pyqtSignal(str)
    def __init__(self, loss_plot):
        super().__init__()
        self.loss_plot = loss_plot

        self.epoch_n=100
        self.train_idx = 1
        self.pred_idx = 2

        self.select_threshold = 0.3
        self.zero_threshold = 0.05

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.target_type = []

    
    def reset(self, 
            TEST_MODE,
            PRETRAIN, 
            TRAIN, 
            target_type, 
            std_IQM,  
            key,
            input_dim, 
            output_dim
        ):

        self.TEST_MODE = TEST_MODE
        self.PRETRAIN = PRETRAIN
        self.TRAIN = TRAIN

        self.target_type = target_type
        self.std_IQM = std_IQM
        self.key = key
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.x_train = []
        self.y_train = []
        self.models = {}
        self.optimizers = {}
        self.criterions = {}

        for t in self.target_type:
            model =  My_Model(input_dim, 1) #.to(self.device)
            self.models[t] = model
            self.optimizers[t] = torch.optim.AdamW(self.models[t].parameters(), lr=1e-4)
            self.criterions[t] = nn.MSELoss(reduction='mean')

        if self.PRETRAIN:
            self.epoch_n=50
            self.pred_idx = 3
            for t in self.target_type:
                path = "myPackage/ML/Model/{}/{}".format(self.key, t)
                if os.path.exists(path):
                    self.models[t].load_state_dict(torch.load(path))
                    self.optimizers[t] = torch.optim.AdamW(self.models[t].parameters(), lr=1e-6)
                    self.log_info_signal.emit("Load pretrain model: {}".format(path))
                else:
                    self.log_info_signal.emit("找不到pretrain model: {}".format(path))
        

    def update_dataset(self, x, y):
        # print(x,y)
        if (np.abs(y)/self.std_IQM>self.select_threshold).any():
            # y[np.abs(y)<self.zero_threshold] = 0
            self.x_train.append(x.tolist())
            self.y_train.append(y.tolist())

    def train(self):
        # bs = 2
        bs = 16
        train_dataset = My_Dataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        
        loss_record = {}
        for t in self.target_type:
            loss_record[t] = []
            self.models[t].train()

        for epoch in range(self.epoch_n):
            for x, y in train_loader:
                for i, t in enumerate(self.target_type):
                    output = self.models[t](x)
                    # print(y.T, output)
                    loss = self.criterions[t](output, y.T[i].reshape(output.shape[0], 1))
                    # Compute gradient(backpropagation).
                    loss.backward()
                    # Update parameters.
                    self.optimizers[t].step()
                    loss_record[t].append(loss.detach().item())
            
            if (epoch+1) % 10 == 0:
                mean_train_loss = []
                for t in self.target_type:
                    mean_train_loss.append(sum(loss_record[t])/len(loss_record[t]))
                # print(mean_train_loss)
                self.loss_plot.update(mean_train_loss)  # plot loss

    def predict(self, x):
        pred = []
        for t in self.target_type:
            self.models[t].eval()
            pred.append(self.models[t](torch.FloatTensor(x.tolist())).detach().numpy()[0])
        # print(pred)
        return pred

    def save_model(self):
        if len(self.target_type)>0:
            with open("dataset.json", "w") as outfile:
                data = {}
                data["x_train"] = list(self.x_train)
                data["y_train"] = list(self.y_train)
                json.dump(data, outfile)
                
            if self.TRAIN:
                for type in self.target_type:
                    path = "myPackage/ML/Model/{}/{}".format(self.key, type)
                    torch.save(self.models[type].state_dict(), path)
                    self.log_info_signal.emit("save model: {}".format(path))
    


    

        