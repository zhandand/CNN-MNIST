import math


n_epochs = 5
Shuffle_dataset = True
random_Seed = 14
supervise_rate = 0.2
train_rate = 0.8
batch_size = 50
num_workers = 0
dataset_size=60000
root = "data\mnist-in-csv\mnist_train.csv"
datapath = "D:\study\Code\python_codes\CNN\data\mnist-in-csv\mnist_train.csv"

indices = list(range(dataset_size))
supervise_split = math.floor(supervise_rate*dataset_size)
train_split = math.floor(train_rate*supervise_split)
validation_split = supervise_split-train_split

train_indices,val_indices,unsup_indices = indices[:train_split],indices[train_split:supervise_split],indices[supervise_split:]
