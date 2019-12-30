from Semi_Supervised.config import *
import torch.utils.data
import torch.nn as nn
import csv
import numpy
import math
import os
import torchvision
import matplotlib.pyplot as plt
from Semi_Supervised.DataMgr import CSVSet

test_datapath = os.getcwd() + "\mnist_in_csv\mnist_test.csv"


def plot_image(inputs):
    img = inputs[:, 0, :, :]
    print(img.size())
    img = img[:, numpy.newaxis, :, :].cpu()
    print(img.size())
    img = torchvision.utils.make_grid(img)

    img = img.detach().numpy().transpose(1, 2, 0)

    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    plt.imshow(img)
    plt.show()


dataset = CSVSet(test_datapath)
testset = torch.utils.data.Subset(dataset, indices=[i for i in list(range(batch_size))])

test_dataloader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=batch_size,
                                              )
for data in test_dataloader:
    conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1).cuda()
    relu = nn.ReLU().cuda()
    conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).cuda()
    pool = nn.MaxPool2d(stride=2, kernel_size=2).cuda()

    image, label = data
    plot_image(image)
    image = conv1(image)
    plot_image(image)

    image = relu(image)

    image = conv2(image)
    plot_image(image)

    print("pooling...")
    image = pool(image)
    plot_image(image)
