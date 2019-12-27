import csv
import pandas as pd

df = pd.read_csv("mnist-in-csv\mnist_train.csv",index_)
def write_data(index,label):
    with open("mnist-in-csv\mnist_train.csv", "r", encoding="utf-8")as f:
        reader = csv.reader(f)
        writer = csv.DictWriter(f,fieldnames="label")
        data =
        writer.writerow({"label":label})

write_data(1,"11")