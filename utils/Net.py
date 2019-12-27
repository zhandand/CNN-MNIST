import torchvision
import torch
import torch.nn as nn
import numpy
import csv
from utils.config import *

# parsing data from csv file
def parse_data():
    with open("mnist-in-csv\mnist_train.csv", "r", encoding="utf-8")as f:
        reader = csv.reader(f)
        rows= [row for row in reader]
        del(rows[0])
    return rows


# get label and img
def get_data(data):
    label = data[0]
    label = int(label)
    img = data[1:]
    img = [pixel.lstrip() for pixel in img ]
    img = list(map(int, img))
    label,img=numpy.array(label),numpy.array(img)
    if use_GPU:
        label ,img = torch.from_numpy(label),torch.from_numpy(img)
    return img.view(1,1,28,28),label

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

whole_data =parse_data()
train_loader = whole_data[:train_split]
validation_loader = whole_data[train_split:supervise_split]
unsup_loader =  whole_data[supervise_split:]



