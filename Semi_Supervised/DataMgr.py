import pandas as pd
from Semi_Supervised.config import *
import torch.utils.data
import csv
import numpy
import math
import os

indices = list(range(dataset_size))
supervise_size = math.floor(supervise_rate * dataset_size)
train_size = math.floor(train_rate * supervise_size)
validation_size = supervise_size - train_size
unsup_size = dataset_size * (1 - supervise_rate)


# 重写dataloader
class CSVSet(torch.utils.data.Dataset):

    def __init__(self, fileUrl, heading=True):
        self.fileUrl = fileUrl

        with open(self.fileUrl, "r", encoding="utf-8")as f:
            self.reader = csv.reader(f)
            self.rows = [row for row in self.reader]
            if heading:
                del (self.rows[0])

        print('Load data from {}'.format(fileUrl))

    def __getitem__(self, index):
        data = self.rows[index]
        label = int(data[0])
        img = data[1:]
        img = [pixel.lstrip() for pixel in img]
        img = list(map(int, img))
        label, img = numpy.array(label), numpy.array(img)
        label, img = torch.from_numpy(label), torch.from_numpy(img)
        label, img = label.cuda(), img.cuda()
        return img.view(1, 28, 28).float(), label.long()

    def __len__(self):
        return len(self.rows)

    # 获取'无标签'的图片
    def getImg(self):
        imgSet = [img[1:] for img in self.rows[supervise_size:]]
        return imgSet

dataset = CSVSet(datapath)

# 按照一定的比例划分数据集
train_indices, val_indices, unsup_indices = indices[:train_size], \
                                            indices[train_size:supervise_size], \
                                            indices[supervise_size:]


# 获取第一轮data_loader
def get_round1_dataloader():
    _round1_train_set = torch.utils.data.Subset(
        dataset=dataset, indices=train_indices)
    _round1_validation_set = torch.utils.data.Subset(
        dataset=dataset, indices=val_indices)

    _round1_train_dataloader = torch.utils.data.DataLoader(
        dataset=_round1_train_set,
        batch_size=batch_size,
        drop_last=True)

    _round1_validation_dataloader = torch.utils.data.DataLoader(
        dataset=_round1_validation_set,
        batch_size=batch_size,
        drop_last=True)
    return _round1_train_dataloader, _round1_validation_dataloader


def get_unlablled_dataloader():
    _unlabelled_dataset = torch.utils.data.Subset(
        dataset=dataset, indices=unsup_indices)
    _unlabelled_dataloader = torch.utils.data.DataLoader(
        dataset=_unlabelled_dataset,
        batch_size=batch_size,
        drop_last=True)
    return _unlabelled_dataloader


# 第二轮data_loader
def get_round2_dataloader():
    _supervise_dataset = torch.utils.data.Subset(
        dataset=dataset, indices=indices[:supervise_size])
    _generate_dataset = CSVSet(
        os.getcwd() + "\mnist_in_csv\mnist_generate.csv", False)
    # 合并监督学习数据集和生成的数据集
    _round2_dataset = torch.utils.data.ConcatDataset(
        [_supervise_dataset, _generate_dataset])
    _round2_train_index, _round2_validaton_index = shuffle_dataset()

    _round2_train_dataset = torch.utils.data.Subset(dataset=_round2_dataset,
                                                    indices=_round2_train_index, )
    _round2_validation_dataset = torch.utils.data.Subset(
        dataset=_round2_dataset, indices=_round2_validaton_index)

    _round2_train_dataloader = torch.utils.data.DataLoader(
        dataset=_round2_train_dataset,
        batch_size=batch_size,
        drop_last=True
    )
    _round2_validation_dataloader = torch.utils.data.DataLoader(
        dataset=_round2_validation_dataset,
        batch_size=batch_size,
        drop_last=True)
    return _round2_train_dataloader, _round2_validation_dataloader


# 打乱数据集
def shuffle_dataset():
    train_index = []
    size = dataset_size * train_rate
    i = 0
    while (1):
        if i == size:
            break
        index = numpy.random.randint(0, dataset_size, dtype=int)
        if index not in train_index:
            train_index.append(index)
            i += 1

    validaton_index = []
    for index in range(dataset_size):
        if index not in train_index:
            validaton_index.append(index)
    return train_index, validaton_index


# 将生成的数据写到文件
def label_to_file(label, img, filepath):
    data_frame = []
    for j, _ in enumerate(label):
        label_part = [label[j]]
        img_part = img[j]
        label_part.extend(img_part)
        data = tuple(label_part)
        data_frame.append(data)
    data = pd.DataFrame(data_frame, index=range(len(label)))
    try:
        data.to_csv(filepath, index=False, header=False)
    except FileNotFoundError:
        print("File name Invalid")
    else:
        print("Parameters are save in path " + filepath)
