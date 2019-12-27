import torchvision
import torch
import torch.nn as nn

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