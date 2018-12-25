# coding=utf-8

import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
import time
import matplotlib.pyplot as plt


# data transform method, standardize data
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

# get train/test data
train_set = MNIST('./data', train=True, transform=data_tf, download=True)
test_set = MNIST('./data', train=False, transform=data_tf, download=True)

# define loss function
criterion = nn.CrossEntropyLoss()

# adapt learning rate for every single gradient
# define audograd
def sgd_adagrad_update(parameters, sqrs, lr):
    eps = 1e-10
    for (param, sqr) in zip(parameters, sqrs):
        sqr[:] = sqr + param.data ** 2
        param.data = param.data - lr * param.grad.data / torch.sqrt(sqr + eps)


def train(num_epochs, batch_size, learning_rate, default_optim=False):
    net = nn.Sequential(
        nn.Linear(28 * 28, 400),
        nn.ReLU(),
        nn.Linear(400, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adagrad(net.parameters(), learning_rate)

    sqrs = []
    for param in net.parameters():
        sqrs.append(torch.zeros_like(param.data))

    losses = []
    start = time.time()
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (inputs, targets) in enumerate(train_data):
            out = net(inputs)
            loss = criterion(out, targets)

            if default_optim:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                net.zero_grad()
                loss.backward()
                sgd_adagrad_update(net.parameters(), sqrs, learning_rate)

            train_loss += loss.data.item()
            if i % 30 == 29:
                losses.append(loss.data.item())
        print("Epoch %s, Train loss: %.6f " % (epoch, train_loss / len(train_data)))
    end = time.time()
    print("It takes %s seconds." % (int(end - start)))
    return losses

train_losses0 = train(num_epochs=5, batch_size=64, learning_rate=1e-2, default_optim=False)
train_losses1 = train(num_epochs=5, batch_size=64, learning_rate=1e-2, default_optim=True)

x_axis = np.linspace(0, 5, len(train_losses0), endpoint=True)
plt.semilogy(x_axis, train_losses0, label='user_defined')
plt.semilogy(x_axis, train_losses1, label='adagrad')
plt.legend(loc='best')
plt.show()
