import csv
import pandas as pd
from utils.Net import *
import os
import time

def attach_label(df,index,label):

    df.loc[index,0] = str(label)


def write_to_file(df):
    df.to_csv(root, index=False, header=False)




model = Model()
model.cuda()
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(torch.load('model_parameter.pkl'))


generate_label = []

for data in unsup_loader:

    img,true_label=data
    outputs = model(img)
    pred = torch.max(outputs.data, 1)[1].cuda().squeeze().cpu()
    generate_label.extend(pred)

print("generate labels finished")


index = supervise_split+1
df = pd.read_csv(root, header=None, low_memory=False)

for index in range(index,dataset_size):
    label = generate_label[index].item()
    attach_label(df,index,label)
    if (index-supervise_split)%500==0:
        print("have labeled the"+str(index-supervise_split))

write_to_file(df)
#     for i in range(batch_size):
#         index += 1
#         label = pred[i].item()
#         true_l = true_label[i].item()
#         print(index,label,true_l)
#         attach_label(df, index, label)

