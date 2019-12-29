from __future__ import division
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
from utils.DataMgr import *

model = Model()
model.cuda()
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(torch.load('D:\study\Code\python_codes\CNN\Pseudo-Labelling\model_parameter.pkl'))

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0.0

    print("Epoch {}/{}".format(epoch + 1, n_epochs))
    print("-" * 10)
    i = 0
    for data in round2_train_dataloader:
        x_train, y_train = data

        outputs = model(x_train)

        pred = torch.max(outputs.data, 1)[1].cuda().squeeze()

        optimizer.zero_grad()

        loss = cost(outputs, y_train)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train).item()
        i += 1
        if i % 10 == 0:
            print("trained " + str(i * batch_size) + " Training Accuracy:{.4f}%".format(
                100 * running_correct / (i * batch_size)))

    testing_correct = 0.0
    for data in round2_validation_dataloader:
        x_test, y_test = data
        outputs = model(x_test)
        pred = torch.max(outputs.data, 1)[1].cuda().squeeze()
        testing_correct += torch.sum(pred == y_test).item()

    print("Loss is:{:.4f}, Train Accuracy is:"
          "{:.4f}%, Test Accuracy is:{:.4f}%".format(100 * running_loss / 48000,
                                                     100 * running_correct / 48000,
                                                     100 * testing_correct / 12000))
    torch.save(model.state_dict(), "model_parameter.pkl")
