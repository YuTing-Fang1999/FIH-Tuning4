
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import os

class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 4, bias=False),
            nn.ReLU(),
            nn.Linear(4, output_dim, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


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

model =  My_Model(32, 1)
# model.load_state_dict(torch.load("myPackage/ML/Model/WNR/TEST"))
            
architecture = [32]
links = []
i=0
# Display all model layer weights
for name, para in model.named_parameters():
    print('{}: {}'.format(name, para.shape))
    architecture.append(para.shape[0])
    for t in range(para.shape[0]):
        for s in range(para.shape[1]):
            source = "{}_{}".format(i, s)
            target = "{}_{}".format(i+1, t)
            id = "{}-{}".format(source, target)
            weight = para[t][s].tolist()

            link = {"id":id, "source":source, "target":target, "weight":weight}
            # print(link)
            links.append(link)
    i+=1

    
ML_data = {}
ML_data["links"] = links
ML_data["architecture"] = architecture
print(architecture)

# Writing to sample.json
with open("ML_data.json", "w") as outfile:
    outfile.write(json.dumps(ML_data, indent=4))