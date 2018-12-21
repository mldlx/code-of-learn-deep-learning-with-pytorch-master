#!/usr/bin/python

import torch

counter1 = torch.tensor([0])
counter2 = torch.tensor([10])

while (counter1 < counter2).item():
    counter1 += 2
    counter2 += 1

print(counter1)
print(counter2)
