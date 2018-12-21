import numpy as np
import torch
from torch import nn


# =============================================================================
# part 1: initialize parameters by numpy
# define a model
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
    nn.ReLU()
)

# the parameters of the 1st layer!
w1 = net1[0].weight
b1 = net1[0].bias

# note: net1[0].weight is Parameter, net1[0].weight.data is Tensor
net1[0].weight.data = torch.from_numpy(np.random.uniform(3, 5, size=(40, 30)))

# initialize model parameters within a loop!
for layer in net1:
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.uniform(0, 0.5, size=param_shape))


# xavier
class SimNet(nn.Module):
    def __init__(self):
        super(SimNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU()
        )

        self.layer1[0].weight.data = torch.randn(40, 30)  # initialize a layer parameters directly!

        self.layer2 = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


net2 = SimNet()
# children method can only access Sequential layer, modules method can access
# not only Sequential layer not also Linear!
for i in net2.children():
    print(i)

for i in net2.modules():
    print(i)

for layer in net2.modules():
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.uniform(0, 0.5, size=param_shape))

# =============================================================================
# part 2: initialize parameters by torch.nn.init
from torch.nn import init

print(net1[0].weight)
# initialize parameters by Xavier method
# [Understanding the difficulty of training deep feedforward neural networks]
init.xavier_uniform(net1[0].weight)

# =============================================================================
# summary:
# above all, introduce two different method to initialize parameters, but in fact,
# they do the same thing, that is change the values of one layer parameters.

# =============================================================================
# question:
# 1. how do the initialization method effect model training, and how much?