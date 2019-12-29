from Semi_Supervised.DataMgr import *
from Semi_Supervised.config import *
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os


class PLModel:

    def __init__(self, _Model, _parametersFileName="model_parameter.pkl", ):
        self.Model = _Model
        self.parametersFileName = _parametersFileName
        self.writer = SummaryWriter(os.getcwd() + "\\visualization_tool\\log")


    def train_round1(self, round1_train_dataloader, round1_validation_dataloader):
        print("Round1 Start...")
        self.train(round1_train_dataloader,
                   round1_validation_dataloader,
                   saveParametersPath=os.getcwd() + "\parameters\\round1_parameters.pkl")
        print("Round1 Finish")
        print("-" * 20)

    def mark(self, unlabbled_data_loader):
        print("Marking...")
        unlabelled_dataloader = unlabbled_data_loader
        generate_label = []

        for data in unlabelled_dataloader:
            img, true_label = data
            outputs = self.Model(img)
            pred = torch.max(outputs.data, 1)[1].cuda().squeeze().cpu()
            generate_label.extend(pred)
        print("Generate labels Finish")

        generate_label = [label.numpy().item() for label in generate_label]
        unlabelled_img = dataset.getImg()
        label_to_file(
            generate_label,
            unlabelled_img,
            filepath=os.getcwd() + "\mnist_in_csv\mnist_generate.csv")
        print("Write to file Finish")
        print("Mark Finish")
        print("-" * 20)

    def train_round2(self, round2_train_dataloader, round2_validation_dataloader):
        print("Round2 Start...")
        self.train(
            round2_train_dataloader,
            round2_validation_dataloader,
            loadParametersPath=os.getcwd() + '\parameters\\round1_parameters.pkl',
            saveParametersPath=os.getcwd() + "\paramters\\round2_paramters.pkl")

    def train(self, train_dataloader, validation_dataloader,
              loadParametersPath=None,
              saveParametersPath=os.getcwd() + "\paramters\\paramters.pkl"):


        cost = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.Model.parameters())
        if loadParametersPath is not None:
            try:
                self.Model.load_state_dict(
                    torch.load(loadParametersPath))
            except FileNotFoundError:
                print("Load parameters Fail!")
            else:
                print("Load paramters from " + loadParametersPath)

        for epoch in range(n_epochs):
            running_loss = 0.0
            running_correct = 0.0

            print("Epoch {}/{}".format(epoch + 1, n_epochs))
            print("-" * 10)
            i = 0
            for data in train_dataloader:
                x_train, y_train = data
                self.writer.add_graph(self.Model, input_to_model=x_train)
                self.writer.add_images(tag='image', img_tensor=x_train, dataformats='NCHW')
                outputs = self.Model(x_train)
                pred = torch.max(outputs.data, 1)[1].cuda().squeeze()

                optimizer.zero_grad()

                loss = cost(outputs, y_train)
                loss.backward()

                optimizer.step()
                running_loss += loss.item()
                running_correct += torch.sum(pred == y_train).item()
                i += 1
                if i % 20 == 0:
                    print("Trained {:d} Training Accuracy:{:.4f}".format
                          (i * batch_size, running_correct / (i * batch_size)))

            j = 0;
            testing_correct = (0.0)
            for data in validation_dataloader:
                j += 1
                x_test, y_test = data
                outputs = self.Model(x_test)
                pred = torch.max(outputs.data, 1)[1].cuda().squeeze()
                testing_correct += torch.sum(pred == y_test).item()
                self.writer.add_scalar('accuracy', testing_correct)

            self.writer.add_scalar('loss', 100 * running_loss / (i * batch_size))
            for i, (name, param) in enumerate(self.Model.named_parameters()):
                self.writer.add_histogram(name, param, 0)
            print(
                "Loss is:{:.4f}, Train Accuracy is:"
                "{:.4f}%, Test Accuracy is:{:.4f}%".format(
                    100 * running_loss / (i * batch_size),
                    100 * running_correct / (i * batch_size),
                    100 * testing_correct / (j * batch_size)))

        try:
            torch.save(self.Model.state_dict(), saveParametersPath)
        except FileNotFoundError:
            print("Save Path Invalid!")
        else:
            print("Save parameters Success!")
