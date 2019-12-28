import csv
import pandas as pd
from utils.Net import *


def attach_label(df,index,label):
    df.loc[index,0] = str(label)

def write_to_file(df):
    df.to_csv(root, index=False, header=False)


model = Model()
model.cuda()
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(torch.load('model_parameter.pkl'))


index = supervise_split
for data in unsup_loader:
    df = pd.read_csv(root, header=None, low_memory=False)
    img,true_label=data
    outputs = model(img)
    pred = torch.max(outputs.data, 1)[1].cuda().squeeze().cpu()

    for write_circle in range(10):
        for i in range(batch_size):
            index += 1
            label = pred[i].item()
            true_l = true_label[i].item()
            print(index,label,true_l)
            attach_label(df, index, label)
    write_to_file(df)



