import pandas as pd
from utils.config import *
import torch.utils.data
import csv
import numpy


indices = list(range(dataset_size))
supervise_size = math.floor(supervise_rate * dataset_size)
train_size = math.floor(train_rate * supervise_size)
validation_size = supervise_size - train_size
unsup_size = dataset_size * (1 - supervise_rate)


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

    def getImg(self):
        imgSet = [img[1:] for img in self.rows[supervise_size:]]
        return imgSet

dataset = CSVSet(datapath)

train_indices, val_indices, unsup_indices = indices[:train_size], \
                                            indices[train_size:supervise_size], \
                                            indices[supervise_size:]

train_set = torch.utils.data.Subset(dataset=dataset, indices=train_indices)
validation_set = torch.utils.data.Subset(dataset=dataset, indices=val_indices)
unsup_set = torch.utils.data.Subset(dataset=dataset, indices=unsup_indices)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           )

validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                batch_size=batch_size,
                                                )

unsup_loader = torch.utils.data.DataLoader(dataset=unsup_set,
                                           batch_size=batch_size,
                                           )

train_sampler1 = torch.utils.data.SubsetRandomSampler(indices[:20000])
validation_sampler1 = torch.utils.data.SubsetRandomSampler(indices[20000:25000])


supervise_dataset = torch.utils.data.Subset(dataset=dataset, indices=indices[:supervise_size])

generate_dataset = CSVSet("D:\study\Code\python_codes\CNN\Pseudo-Labelling\mnist_generate.csv",
                          False)

round2_dataset = torch.utils.data.ConcatDataset([supervise_dataset, generate_dataset])


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


train_index, validaton_index = shuffle_dataset()

round2_train_dataset = torch.utils.data.Subset(dataset=round2_dataset,
                                               indices=train_index)
round2_validation_dataset = torch.utils.data.Subset(dataset=round2_dataset,
                                                    indices=validaton_index)

round2_train_dataloader = torch.utils.data.DataLoader(dataset=round2_train_dataset,
                                                      batch_size=batch_size)
round2_validation_dataloader = torch.utils.data.DataLoader(dataset=round2_validation_dataset,
                                                           batch_size=batch_size)

def label_to_file(label, img, generate_file):
    data_frame = []
    for j, _ in enumerate(label):
        label_part = [label[j]]
        img_part = img[j]
        label_part.extend(img_part)
        data = tuple(label_part)
        data_frame.append(data)
    data = pd.DataFrame(data_frame, index=range(len(label)))
    data.to_csv(generate_file, index=False, header=False)

