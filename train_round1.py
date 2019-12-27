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
import math
import csv


transform = transforms.Compose([
     transforms.ToTensor(),

    transforms.Normalize([0.5],[0.5])])


whole_data =parse_data()
train_loader = whole_data[:train_split]
validation_loader = whole_data[train_split:supervise_split]
unsup_loader =  whole_data[supervise_split:]

model = Model()
model.cuda()
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(torch.load('model_parameter.pkl'))


for epoch in range(n_epochs):
    running_loss =0.0
    running_correct = 0.0

    print("Epoch {}/{}".format(epoch+1,n_epochs))
    print("-"*10)
    i = 0
    for data in train_loader:
        x_train,y_train = get_data(data)
        x_train,y_train=x_train.cuda().float(),y_train.cuda().long()
        y_train = y_train.unsqueeze(0)
        # x_train,y_train = Variable(x_train),Variable(y_train)

        outputs = model(x_train)

        pred =torch.max(outputs.data,1)[1].cuda().squeeze()

        optimizer.zero_grad()

        loss = cost(outputs,y_train.long())

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
        running_loss+=loss.item()
        running_correct +=torch.sum(pred==y_train.item()).long()
        i += 1
        if i == 500:
            print("Example" + str(i) + " Running correct:"+str(running_correct))


    testing_correct = (0.0)
    for data in validation_loader:
        x_test,y_test = get_data(data)
        x_test,y_test=x_test.cuda().float(),y_test.cuda().long()
        y_test = y_test.unsqueeze(0)
        outputs = model(x_test)
        pred = torch.max(outputs.data,1)[1].cuda().squeeze()
        testing_correct += torch.sum(pred==y_test.item()).long()

    print("Loss is:{:.4f}, Train Accuracy is:"
          "{:.4f}%, Test Accuracy is:{:.4f}%".format(running_loss / train_split,
                        100.0* running_correct.item() / (train_split*1.0),100.0*testing_correct.item()/(validation_split*1.0)))
    torch.save(model.state_dict(), "model_parameter.pkl")

X_test, y_test = next(iter(validation_loader))
X_test,y_test = X_test.cuda(),y_test.cuda()

pred = model(X_test)
pred = torch.max(pred, 1)[1].cuda().squeeze()

print("Predict Label is:", [ i for i in pred])
print("Real Label is:",[i for i in y_test])

X_test =X_test.cpu()
img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
plt.imshow(img)
plt.show()




