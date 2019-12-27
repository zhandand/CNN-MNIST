import csv
import pandas as pd


def attach_label(df,index,label):
    root = "mnist-in-csv\mnist_train.csv"
    df = pd.read_csv(root, header=None)
    print(df)
    df.loc[1:1,index] = [str(label)]
    df.to_csv(root, index=False, header=False)
    df = pd.read_csv(root, header=None)
    print(df)

attach_label(0,5)