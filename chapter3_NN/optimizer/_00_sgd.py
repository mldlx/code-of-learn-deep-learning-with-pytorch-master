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

# gradient decent method function: \theta_{i+1} = \theta_i - \eta \nabla L(\theta)
def sgd_update(parameters, lr):
    for param in parameters:
        param.data = param.data - lr * param.grad.data


def train(epoch_nums, batch_size, learning_rate, default_optim=False):
    print('=' * 80)
    # define net
    net = nn.Sequential(
        nn.Linear(28 * 28, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), learning_rate)

    losses = []
    start = time.time()
    for epoch in range(epoch_nums):
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
                sgd_update(net.parameters(), learning_rate)

            train_loss += loss.data.item()
            if i % 30 == 29:
                losses.append(loss.data.item())
        print("Epoch %s, Train loss: %.6f" % (epoch, train_loss/len(train_data)))
    end = time.time()
    print('it takes %.1f seconds.' % (end - start))

    x_axis = np.linspace(0, epoch_nums, len(losses), endpoint=True)
    plt.semilogy(x_axis, losses, label='batch_size=%s' % batch_size)
    plt.legend(loc='best')
    plt.show()


train(epoch_nums=5, batch_size=1, learning_rate=1e-2, default_optim=False)
train(epoch_nums=5, batch_size=64, learning_rate=1e-2, default_optim=False)
train(epoch_nums=5, batch_size=64, learning_rate=1, default_optim=False)
train(epoch_nums=5, batch_size=1, learning_rate=1e-2, default_optim=True)
train(epoch_nums=5, batch_size=64, learning_rate=1e-2, default_optim=True)
