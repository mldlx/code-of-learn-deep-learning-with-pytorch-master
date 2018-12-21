import numpy as np
import torch
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


def data_transforam(x):
    x = np.array(x, dtype=np.float32) / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1, ))
    x = torch.from_numpy(x)
    return x


train_data = mnist.MNIST('./data', train=True, transform=data_transforam, download=True)
test_data = mnist.MNIST('./data', train=False, transform=data_transforam, download=True)


train_data = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = DataLoader(test_data, batch_size=128, shuffle=False)


# different between functional and sequential
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(28*28, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, 100)
        self.linear4 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


net = Net()

'''
net2 = nn.Sequential(
    nn.Linear(28 * 28, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
'''



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc, train_num = 0, 0, 0
    for i, (inputs, targets) in enumerate(train_data):

        # forward
        out = net(inputs)
        loss = criterion(out, targets)

        # zero the parameter gradients & backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        _, pred = out.max(1)
        train_acc += (pred == targets).sum().item()
        train_num += inputs.shape[0]

    test_loss, test_acc, test_num = 0, 0, 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_data):
            out = net(inputs)
            loss = criterion(out, targets)
            test_loss += loss.item()
            _, pred = out.max(1)
            test_acc += (pred == targets).sum().item()
            test_num += inputs.shape[0]

    print("Epoch[%s] Train Loss: %.6f, Train Acc: %.6f; Test Loss: %.6f, Test Acc: %.6f."
          % (epoch, train_loss/train_num, train_acc/train_num, test_loss/test_num, test_acc/test_num))
