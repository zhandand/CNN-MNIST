from __future__ import division
import torchvision
import torch
import numpy
import random
from torchvision import datasets,transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from torch.autograd import Variable
# from Net import Model
import math
import torch.nn.functional as F
import csv

n_epochs = 5
Shuffle_dataset = True
random_Seed = 14
supervise_rate = 0.2
train_rate = 0.8
batch_size = 1
num_workers = 0
dataset_size=60000
use_GPU = True

transform = transforms.Compose([
     transforms.ToTensor(),

    transforms.Normalize([0.5],[0.5])])

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
    img = list(map(float, img))
    label,img=numpy.array(label),numpy.array(img)
    if use_GPU:
        label ,img = torch.from_numpy(label),torch.from_numpy(img)
    return img.view(1,1,28,28),label

indices = list(range(dataset_size))
supervise_split = math.floor(supervise_rate*dataset_size)
train_split = math.floor(train_rate*supervise_split)
validation_split = supervise_split-train_split

train_indices,val_indices,unsup_indices = indices[:train_split],indices[train_split:supervise_split],indices[supervise_split:]


# train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
# validation_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
# unsup_sampler  =torch.utils.data.SubsetRandomSampler(unsup_indices)


# train_loader = torch.utils.data.DataLoader(dataset=M,
#                                            batch_size=batch_size,
#                                            sampler=train_sampler,
#                                            num_workers = num_workers)
# validation_loader = torch.utils.data.DataLoader(dataset=M,
#                                                 batch_size=batch_size,
#                                                 sampler=validation_sampler,
#                                                 num_workers=num_workers)
# unsup_loader = torch.utils.data.DataLoader(dataset=M,
#                                            batch_size=batch_size,
#                                            sampler=unsup_sampler,
#                                            num_workers=num_workers)
whole_data =parse_data()
train_loader = whole_data[:train_split]
validation_loader = whole_data[train_split:supervise_split]
unsup_loader =  whole_data[supervise_split:]


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
                                  torch.nn.Linear(1024,10),
                                  )

    def forward(self,x):
        x= self.conv1(x)
        x = x.view(-1,14*14*128)
        x =self.dense(x)
        return x


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




