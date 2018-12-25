

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open(r'D:\Project\code-of-learn-deep-learning-with-pytorch-master\chapter4_CNN\cat.png').convert('L')
image = np.array(image, dtype='float32')
plt.imshow(image.astype('uint8'), cmap='gray')
plt.show()

# batch, in_channels, width, height
inputs = torch.from_numpy(image.reshape(1, 1, image.shape[0], image.shape[1]))
inputs.requires_grad_(True)

# nn.Conv2d

# in_channels, out_channels, kernel_size, stride, padding
conv1 = nn.Conv2d(1, 1, 3, bias=False)
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))

conv1.weight.data = torch.from_numpy(kernel)

edge = conv1(inputs)
edge = edge.data.squeeze().numpy()

plt.imshow(edge, cmap='gray')
plt.show()

weight = torch.from_numpy(kernel)
edge = F.conv2d(inputs, weight)
edge = edge.data.squeeze().numpy()

plt.imshow(edge, cmap='gray')
plt.show()

# ------------------------------------------
pool = nn.MaxPool2d(2, 2)  # kernel_size, stride
print('before max pool image shape: (%s,%s)' % (inputs.shape[2], inputs.shape[3]))
out = pool(inputs).data.squeeze().numpy()
print('before max pool image shape: (%s,%s)' % (out.shape[0], out.shape[1]))
plt.imshow(out, cmap='gray')
plt.show()

out = F.max_pool2d(inputs, 2, 2).data.squeeze().numpy()
print('before max pool image shape: (%s,%s)' % (inputs.shape[2], inputs.shape[3]))
print('before max pool image shape: (%s,%s)' % (out.shape[0], out.shape[1]))
plt.imshow(out, cmap='gray')
plt.show()







