import torchvision
import torch
import numpy
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from utils.config import *
from utils.Net import *
import pandas as pd


class CSVSet(torch.utils.data.DataLoader):

    def __init__(self,fileUrl):
        self.fileUrl = fileUrl


        with open(fileUrl, "r", encoding="utf-8")as f:
            self.reader = csv.reader(f)
            self.rows = [row for row in self.reader]
            del (self.rows[0])

        print('Load data from {}'.format(fileUrl))


    def __getitem__(self, index):

        data = self.rows[index]
        label = data[0]
        label = int(label)
        img = data[1:]
        img = [pixel.lstrip() for pixel in img]
        img = list(map(int, img))
        label, img = numpy.array(label), numpy.array(img)
        label, img = torch.from_numpy(label), torch.from_numpy(img)
        label,img=  label.cuda(),img.cuda()
        return img.view(1,28,28).float(), label.long()


    def __len__(self):
        return len(self.rows)