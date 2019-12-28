from utils.DataMgr import *
import torch.nn as nn


class PLModel:

    def __init__(self, _round1_train_dataloader,
                 _round1_validation_dataloader,
                 _round2_train_dataloader,
                 _round2_validation_dataloader,
                 _Model,
                 _parametersFileName="model_parameter.pkl",
                 _generateFileName="mnist_generate.csv"):
        self.Model = _Model
        self.parametersFileName = _parametersFileName
        self.generateFileName = _generateFileName
        self.round1_train_dataloader = _round1_train_dataloader
        self.round1_validation_dataloader = _round1_validation_dataloader
        self.round2_train_dataloader = _round2_train_dataloader
        self._round2_validation_dataloader = _round2_validation_dataloader

    def train_round1(self, parametersName="round1_parameters"):
        self.train(self.round1_train_dataloader, round2_validation_dataset,
                   saveParametersPath='round1_parameters.pkl')

    def mark(self):
        unlabelled_dataloader = unsup_loader
        generate_label = []

        for data in unlabelled_dataloader:
            img, true_label = data
            outputs = self.Model(img)
            pred = torch.max(outputs.data, 1)[1].cuda().squeeze().cpu()
            generate_label.extend(pred)
        print("generate labels finished")

        generate_label = [label.numpy().item() for label in generate_label]
        unlabel_img = dataset.getImg()
        label_to_file(generate_label, unlabel_img, generate_file=self.generatePath)
        print("write to file finished")

    def train_round2(self):
        self.train(round2_train_loader, round2_validation_dataset,
                   loadParametersPath='D:\study\Code\python_codes\CNN\Pseudo-Labelling\model_parameter.pkl',
                   saveParametersPath='round2_parameters.pkl')

    def train(self, train_dataloader, validation_dataloader,
              loadParametersPath=None,
              saveParametersPath="paramters.pkl"):

        cost = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.Model.parameters())
        if loadParametersPath != None:
            self.Model.load_state_dict(
                torch.load(loadParametersPath))

        for epoch in range(n_epochs):
            running_loss = 0.0
            running_correct = 0.0

            print("Epoch {}/{}".format(epoch + 1, n_epochs))
            print("-" * 10)
            i = 0
            for data in round2_train_loader:
                x_train, y_train = data

                outputs = self.Model(x_train)

                pred = torch.max(outputs.data, 1)[1].cuda().squeeze()

                optimizer.zero_grad()

                loss = cost(outputs, y_train)

                try:
                    loss.backward()
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception

                optimizer.step()
                running_loss += loss.item()
                running_correct += torch.sum(pred == y_train).long()
                i += 1
                if i % 10 == 0:
                    print("trained " + str(i * batch_size) + " Training Accuracy:" + str(
                        running_correct.item() / (i * batch_size)))

            testing_correct = (0.0)
            for data in round2_validation_dataset:
                x_test, y_test = data
                outputs = self.Model(x_test)
                pred = torch.max(outputs.data, 1)[1].cuda().squeeze()
                testing_correct += torch.sum(pred == y_test)

            print("Loss is:{:.4f}, Train Accuracy is:"
                  "{:.4f}%, Test Accuracy is:{:.4f}%".format(100 * running_loss / train_size,
                                                             100 * running_correct.item() / (train_size),
                                                             100 * testing_correct.item() / (validation_size)))
            torch.save(self.Model.state_dict(), saveParametersPath)

    def run(self):
        self.train_round1()
        self.mark()
        self.train_round2()
