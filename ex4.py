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

import torch.nn.functional as functional
import torch.cuda
from torch import nn, optim
from torch.autograd import Variable


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self._conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        self._conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2)
        self._conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=5, stride=1, padding=2)
        self._conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=5, stride=1, padding=2)
        self._conv5 = nn.Conv2d(in_channels=200, out_channels=400, kernel_size=5, stride=1, padding=2)

        self._dropout = nn.Dropout(p=0.5)
        self._l1 = nn.Linear(400 * 5 * 3, 30)
        self._loss_function = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self.parameters(), lr=0.0005)

    def forward(self, x):
        x = Variable(x)
        if torch.cuda.is_available():
            x = x.cuda()
        x = self._conv1(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self._dropout(x)
        x = self._conv2(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self._dropout(x)
        x = self._conv3(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self._dropout(x)
        x = self._conv4(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self._dropout(x)
        x = self._conv5(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self._dropout(x)
        x = x.view(x.size(0), -1)
        return self._l1(x)

    def train_example(self, vectors_batch, labels_batch):
        vectors_batch = Variable(vectors_batch)
        labels_batch = Variable(labels_batch)
        if torch.cuda.is_available():
            vectors_batch = vectors_batch.cuda()
            labels_batch = labels_batch.cuda()
        self._optimizer.zero_grad()
        ys = self(vectors_batch)
        loss = self._loss_function(ys, labels_batch)
        loss.backward()
        self._optimizer.step()


class Ex4:
    def __init__(self, batch_train, batch_test, epoch, acc_check):
        self.batch_train = batch_train
        self.batch_test = batch_test
        self.epoch = epoch
        self.acc_check = acc_check

    def main(self):
        model = MyNet()
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
            for vector_batch, label_batch in tqdm.tqdm(train_loader, total=len(train_loader),
                                                       unit_scale=self.batch_train):
                model.train_example(vector_batch, label_batch)
            model.eval()
            print(self.evaluate(model, train_loader))
            print(self.evaluate(model, validation_loader))

        model.eval()
        all_predictions = []
        for vectors_batch, _ in test_loader:
            outputs = model(vectors_batch)
            _, predictions = torch.max(outputs.data, 1)
            all_predictions.extend(predictions)

        with open("test_y", "w") as f:
            for spect, prediction in zip(gc_test_loader.spects, all_predictions):
                f.write("{}, {}".format(os.path.basename(spect[0]), str(prediction.item())))
                f.write(os.linesep)

    def evaluate(self, model, loader):
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
    Ex4(batch_train=100, batch_test=100, epoch=20, acc_check=5000)
    Ex4.main()
