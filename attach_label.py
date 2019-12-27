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

df = pd.read_csv(root, header=None)

i=1
for data in unsup_loader:
    img,label=get_data(data)
    img,label=img.cuda().float(),label.cuda().long()
    label = label.unsqueeze(0)
    outputs = model(img)
    pred = torch.max(outputs.data, 1)[1].cuda().squeeze()
    attach_label(df,supervise_split+i,pred.item())
    if i%1000==0:
        write_to_file(df)
        df = pd.read_csv(root, header=None)

write_to_file(df)

