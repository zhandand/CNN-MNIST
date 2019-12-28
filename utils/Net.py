import torchvision
import torch
import torch.nn as nn
import numpy
import csv
from utils.config import *
from utils.CSV_Set import CSVSet
import torch.utils.data

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense =nn.Sequential(nn.Linear(14*14*128,1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  torch.nn.Linear(1024,10))

    def forward(self,x):
        x= self.conv1(x)
        x = x.view(-1,14*14*128)
        x =self.dense(x)
        return x


train_indices,val_indices,unsup_indices = indices[:train_split],indices[train_split:supervise_split],indices[supervise_split:]


train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
validation_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
unsup_sampler  =torch.utils.data.SubsetRandomSampler(unsup_indices)

dataset = CSVSet(datapath)

train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler=validation_sampler)
unsup_loader =  torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler=unsup_sampler)



