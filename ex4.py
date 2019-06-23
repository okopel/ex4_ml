#################################
# Ori Kopel, Shlomo Rabonovich  #
# 205533151, 308432517          #
#################################

import os

import torch.cuda
import torch.utils.data
import tqdm
from torch.autograd import Variable
from gcommand_loader import GCommandLoader
import matplotlib.pyplot as plt
import torch.nn.functional as functional
import torch.cuda
from torch import nn, optim
from torch.autograd import Variable


class MyNet(nn.Module):
    def __init__(self, learning_rate, dropOut, classes, linearSize):
        super(MyNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self._dropout = nn.Dropout(p=dropOut)
        self._l1 = nn.Linear(linearSize, classes)
        self._loss_function = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = Variable(x)
        if torch.cuda.is_available():
            x = x.cuda()
        x = self.layer1(x)
        x = self._dropout(x)
        x = self.layer2(x)
        x = self._dropout(x)
        x = self.layer3(x)
        x = self._dropout(x)
        x = self.layer4(x)
        x = self._dropout(x)
        x = x.view(x.size(0), -1)
        return self._l1(x)

    def train_exam(self, vec_batch, label_batch):
        vec_batch = Variable(vec_batch)
        label_batch = Variable(label_batch)
        if torch.cuda.is_available():
            vec_batch = vec_batch.cuda()
            label_batch = label_batch.cuda()
        self._optimizer.zero_grad()
        loss = self._loss_function(self(vec_batch), label_batch)
        loss.backward()
        self._optimizer.step()


class Ex4:
    def __init__(self, batch_train, batch_test, epoch, acc_check, lr, dropout, linearSize, classes):
        self.batch_train = batch_train
        self.batch_test = batch_test
        self.epoch = epoch
        self.acc_check = acc_check
        self.learning_rate = lr
        self.dropout = dropout
        self.linear_size = linearSize
        self.classes = classes

    def main(self):
        plt.title('Loss calculating')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        loss_train_list = []
        loss_valid_list = []
        model = MyNet(learning_rate=self.learning_rate, dropOut=self.dropout, classes=self.classes,
                      linearSize=self.linear_size)
        if torch.cuda.is_available():
            model.cuda()

        train_loader = torch.utils.data.DataLoader(GCommandLoader('./ML4_dataset/data/train'),
                                                   batch_size=self.batch_train, shuffle=True, num_workers=4,
                                                   pin_memory=True, sampler=None)

        validation_loader = torch.utils.data.DataLoader(GCommandLoader('./ML4_dataset/data/valid'),
                                                        batch_size=self.batch_test,
                                                        shuffle=True,
                                                        num_workers=4, pin_memory=True, sampler=None)

        gc_test_loader = GCommandLoader('./ML4_dataset/data')
        test_loader = torch.utils.data.DataLoader(gc_test_loader, batch_size=self.batch_test, shuffle=False,
                                                  num_workers=4, pin_memory=True, sampler=None)

        for i in range(self.epoch):
            print("{}/{}".format(i + 1, self.epoch))
            model.train()
            for vec_batch, lab_batch in tqdm.tqdm(train_loader, total=len(train_loader), unit_scale=self.batch_train):
                model.train_exam(vec_batch, lab_batch)
            model.eval()
            print(self.valuate(model, train_loader))
            print(self.valuate(model, validation_loader))

        model.eval()
        all_predictions = []
        for vec_batch, _ in test_loader:
            output = model(vec_batch)
            _, predictions = torch.max(output.data, 1)
            all_predictions.extend(predictions)

        with open("test_y", "w") as f:
            for spect, prediction in zip(gc_test_loader.spects, all_predictions):
                f.write("{}, {}".format(os.path.basename(spect[0]), str(prediction.item())))
                f.write(os.linesep)
                loss_train_list.append(prediction.item())
                loss_valid_list.append(os.linesep)
        plt.plot(loss_train_list, label="train")
        plt.plot(loss_valid_list, label="valid")
        plt.xticks(range(1, self.epoch))
        plt.legend()
        plt.show()

    def valuate(self, model, loader):
        total = 0
        correct = 0

        for vectors, labels in loader:
            labels = Variable(labels)
            if torch.cuda.is_available():
                labels = labels.cuda()
            outputs = model(vectors)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total >= self.acc_check:
                return correct / total


if __name__ == '__main__':
    ex4 = Ex4(batch_train=100, batch_test=100, epoch=10, acc_check=5000, lr=0.0005, dropout=0.5, linearSize=15360,
              classes=30)
    ex4.main()
