
import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
import time
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self, ):
        super(SimpleCNN, self).__init__()  # b 1 28 28
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),  # b 16 26 26
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3), # b 32 24 24
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # b 32 12 12
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # b 64 10 10
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),  # b 128 8 8
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # b 128 4 4
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x


train_set = MNIST('./data', train=True, transform=data_tf, download=True)
test_set = MNIST('./data', train=False, transform=data_tf, download=True)

net = SimpleCNN()
criterion = nn.CrossEntropyLoss()


def train(num_epochs, batch_size, learning_rate):
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adadelta(net.parameters(), learning_rate)

    losses = []
    start = time.time()
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (inputs, targets) in enumerate(train_data):
            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)

            out = net(inputs)
            loss = criterion(out, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 30 == 29:
                losses.append(loss.data.item())
            train_loss += loss.data.item()
        print("Epoch %s, Train loss: %.6f" % (epoch, train_loss/len(train_data)))
    end = time.time()
    print("it takes %s seconds." % (int(end - start)))
    return losses


train_losses0 = train(num_epochs=5, batch_size=64, learning_rate=1e-2)

x_axis = np.linspace(0, 5, len(train_losses0), endpoint=True)
plt.semilogy(x_axis, train_losses0, label='learning_rate=1e-2')
plt.legend(loc='best')
plt.show()
