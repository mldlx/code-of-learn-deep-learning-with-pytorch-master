import torch
from torch.autograd.variable import Variable

# -----------------------------------------------
# create Variable
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# build a computational graph
y = x * w + b

# compute gradient
y.backward()

# print out the gradients
print(x.grad)
print(w.grad)
print(b.grad)
# -----------------------------------------------
x = torch.tensor([1], requires_grad=True)
w = torch.tensor([2], requires_grad=True)
b = torch.tensor([3], requires_grad=True)

x = torch.Tensor([1]); x.requires_grad_(True)
w = torch.Tensor([2]); w.requires_grad_(True)
b = torch.Tensor([3]); b.requires_grad_(True)

y = w * x + b
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)
# -----------------------------------------------
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = 2 * x
y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)
# ---------------------------------------
from torch.utils.data import DataLoader

class myDataset(DataLoader):
    def __init__(self, csv_file, txt_file, root_dir, other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        data = (self.csv_data[idx], self.txt_data[idx])
        return data

# ---------------------------------------
import torch.utils.data as data_utils

features = torch.randn([10, 5])
targets = torch.randint(low=0, high=2, size=(10, ))
train = data_utils.TensorDataset(features, targets)
train_loader = data_utils.DataLoader(train, batch_size=2, shuffle=True)

for i, batch in enumerate(train_loader):
    print(i, batch)
# ---------------------------------------
import numpy as np

x = np.random.random(100) * 10
y = x * 3 + np.random.uniform(0, 1, 100)
x_train = np.array([[i] for i in x])
y_train = np.array([[i] for i in y])

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.Linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.Linear(x)
        return out

model = LinearRegression().double()

criterion = nn.MSELoss()
optimalizer = optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 100
for epoch in range(num_epochs):

    inputs = Variable(x_train)
    targets = Variable(y_train)

    out = model(inputs)
    loss = criterion(out, targets)

    optimalizer.zero_grad()
    loss.backward()
    optimalizer.step()
    if (epoch + 1) % 20 == 0:
        print("Epoch [%s/%s], loss:%.6f" % (epoch+1, num_epochs, loss.item()))
# ----------------------------------------------
x = torch.randn(10)

def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1, 4)], 1)

make_features(x)






