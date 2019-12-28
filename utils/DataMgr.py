import pandas as pd
from utils.config import *
from utils.CSV_Set import *
import torch.utils.data

indices = list(range(dataset_size))
supervise_size = math.floor(supervise_rate * dataset_size)
train_size = math.floor(train_rate * supervise_size)
validation_size = supervise_size - train_size
unsup_size = dataset_size * (1 - supervise_rate)

train_indices, val_indices, unsup_indices = indices[:train_size], indices[train_size:supervise_size], indices[
                                                                                                      supervise_size:]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
validation_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
unsup_sampler = torch.utils.data.SubsetRandomSampler(unsup_indices)

dataset = CSVSet(datapath)

# TODO:concat the dataloader
# loader = torch.utils.data.ConcatDataset()

train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                sampler=validation_sampler)

unsup_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler=unsup_sampler)


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
