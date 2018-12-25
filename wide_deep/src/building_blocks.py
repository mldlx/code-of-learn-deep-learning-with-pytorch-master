from __future__ import print_function
import numpy as np
import pandas as pd
from wide_deep.data_utils import prepare_data

# -----------------------------------------------------------------------------
# prepare data
DF = pd.read_csv('data/adult_data.csv')
DF['income_label'] = (DF["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

wide_cols = ['age','hours_per_week','education', 'relationship','workclass',
             'occupation','native_country','gender']
crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])
embeddings_cols = [('education',10), ('relationship',8), ('workclass',10),
                    ('occupation',10),('native_country',10)]
continuous_cols = ["age","hours_per_week"]
target = 'income_label'
method = 'logistic'

wd_dataset = prepare_data(DF, wide_cols,crossed_cols,embeddings_cols,continuous_cols,target,scale=True)
# -----------------------------------------------------------------------------
# the model -- wide part
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Wide(nn.Module):
    def __init__(self, wide_dim, n_class):
        super(Wide, self).__init__()
        self.wide_dim = wide_dim
        self.n_class = n_class
        self.linear = nn.Linear(self.wide_dim, self.n_class)

    def forward(self, X):
        out = F.sigmoid(self.linear(X))
        return out


wide_dim = wd_dataset['train_dataset'].wide.shape[1]
n_class = 1
wide_model = Wide(wide_dim, n_class)

train_dataset = np.hstack([wd_dataset['train_dataset'].labels.reshape(-1, 1), wd_dataset['train_dataset'].wide])


optimizer = torch.optim.Adam(wide_model.parameters())
batch_size = 64
n_epochs = 10
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    total = 0
    correct = 0
    for i, batch in enumerate(train_loader):
        X_w = Variable(batch[:, 1:]).float()
        y = Variable(batch[:, 0]).float()
        optimizer.zero_grad()
        y_pred = wide_model(X_w)
        loss = F.binary_cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()

        total+= y.size(0)
        y_pred_cat = (y_pred > 0.5).squeeze(1).float()
        correct+= float((y_pred_cat == y).sum().data[0])
    print('Epoch {} of {}, Loss: {}, accuracy: {}'.format(epoch+1, n_epochs, round(loss.data[0],3), round(correct/total,4)))



for epoch in range(500):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        inputs, labels = batch[:, 1:].float(), batch[:, 0].float()
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = wide_model(inputs)
        loss = F.binary_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
