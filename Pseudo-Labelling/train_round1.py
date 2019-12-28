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
    running_loss =0.0
    running_correct = 0.0

    print("Epoch {}/{}".format(epoch+1,n_epochs))
    print("-"*10)
    i = 0
    for data in train_loader:
        x_train, y_train = data

        outputs = model(x_train)

        pred = torch.max(outputs.data, 1)[1].cuda().squeeze()

        optimizer.zero_grad()

        loss = cost(outputs, y_train)

        try:
            loss.backward()
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train).long()
        i += 1
        if i % 10 == 0:
            print("trained " + str(i * batch_size) + " Training Accuracy:" + str(
                running_correct.item() / (i * batch_size)))

    testing_correct = (0.0)
    for data in validation_loader:
        x_test,y_test = data
        outputs = model(x_test)
        pred = torch.max(outputs.data,1)[1].cuda().squeeze()
        testing_correct += torch.sum(pred == y_test)

    print("Loss is:{:.4f}, Train Accuracy is:"
          "{:.4f}%, Test Accuracy is:{:.4f}%".format(100 * running_loss / train_size,
                                                     100 * running_correct.item() / (train_size),
                                                     100 * testing_correct.item() / (validation_size)))
    torch.save(model.state_dict(), "model_parameter.pkl")

    testing_correct = (0.0)
    for data in unsup_loader:
        x_test, y_test = data
        outputs = model(x_test)
        pred = torch.max(outputs.data, 1)[1].cuda().squeeze()
        testing_correct += torch.sum(pred == y_test)
    print(unsup_size)
    print("Unsup Accuracy is:{:.4f}%".format(
        100 * testing_correct.item() / unsup_size))

# X_test, y_test = next(iter(validation_loader))
# X_test,y_test = X_test.cuda(),y_test.cuda()
#
# pred = model(X_test)
# pred = torch.max(pred, 1)[1].cuda().squeeze()
#
# print("Predict Label is:", [ i for i in pred])
# print("Real Label is:",[i for i in y_test])
#
# X_test =X_test.cpu()
# img = torchvision.utils.make_grid(X_test)
# img = img.numpy().transpose(1,2,0)
#
# std = [0.5,0.5,0.5]
# mean = [0.5,0.5,0.5]
# img = img*std+mean
# plt.imshow(img)
# plt.show()
