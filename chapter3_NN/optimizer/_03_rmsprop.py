import numpy as np
import torch
from torchvision.datasets import MNIST # 导入 pytorch 内置的 mnist 数据
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

train_set = MNIST('./data', train=True, transform=data_tf, download=True)
test_set = MNIST('./data', train=False, transform=data_tf, download=True)

criterion = nn.CrossEntropyLoss()


def train(num_epochs, batch_size, learning_rate, alpha, default_optim):
    net = nn.Sequential(
        nn.Linear(28*28, 400),
        nn.ReLU(),
        nn.Linear(400, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.RMSprop(net.parameters(), learning_rate, alpha)

    sqrs = []
    for param in net.parameters():
        sqrs.append(torch.zeros_like(param.data))

    def sgd_rmsprop(parameters, learning_rate, alpha, sqrs):
        eps = 1e-10
        for (param, sqr) in zip(parameters, sqrs):
            sqr[:] = alpha * sqr + (1 - alpha) * param.data ** 2
            param.data = param.data - learning_rate * param.grad.data / torch.sqrt(sqr + eps)

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
                sgd_rmsprop(net.parameters(), learning_rate, alpha, sqrs)

            if i % 30 == 29:
                losses.append(loss.data.item())
            train_loss += loss.data.item()
        print("Epoch %s, Train loss: %.6f" % (epoch, train_loss/len(train_data)))
    end = time.time()
    print("it takes %s seconds." % (int(end - start)))
    return losses


train_losses0 = train(num_epochs=5, batch_size=64, learning_rate=1e-2, alpha=0.9, default_optim=False)
train_losses1 = train(num_epochs=5, batch_size=64, learning_rate=1e-2, alpha=0.9, default_optim=True)

x_axis = np.linspace(0, 5, num=len(train_losses0))
plt.plot(x_axis, train_losses0, label='user_defined')
plt.plot(x_axis, train_losses0, label='default')
plt.legend(loc='best')
plt.show()
