# Import library
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets

# Set hyperparameters for training
learning_rate = 1e-3
batch_size = 64

# Number of subprocesses to use for data loading
num_workers = 0

# Create directories
ckpt_dir = './checkpoint'
log_dir = './log'

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


## Save & Load network
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               './%s/model_epoch%d.pth' % (ckpt_dir, epoch))


def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim


## Load MNIST dataset

# Convert data to torch.FloatTensor
transform = transforms.ToTensor()

# MNIST: 28*28 dimension -> 784 length vectors
mnist_test = datasets.MNIST(root='./data', train=True, download=False, transform=transform)

loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

num_data = len(loader.dataset)
num_batch = np.ceil(num_data / batch_size)

print('# of data:', num_data)
print('Size of batch:', num_batch)

## Create AutoEncoder network
class AutoEncoder(nn.Module):
    def __init__(self):
        encoding_dim = 20
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Linear(28 * 28, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, 28 * 28)

    def forward(self, x):
        x = x.view(batch_size, -1)
        encoded = torch.relu(self.encoder(x))
        out = torch.sigmoid(self.decoder(encoded).view(batch_size, 1, 28, 28))
        return out


## Loss function & Optimizer
net = AutoEncoder()
print(net)

fn_loss = nn.MSELoss()
fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Log
writer = SummaryWriter(log_dir=log_dir)

net, optim = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

## TESTING

with torch.no_grad():
    net.eval()
    loss_arr = []
    acc_arr = []

    for batch, [image, label] in enumerate(loader, 1):
        x = image

        output = net.forward(x)
        pred = fn_pred(output)

        loss = fn_loss(output, x)
        acc = fn_acc(pred, x)

        loss_arr += [loss.item()]
        acc_arr += [acc.item()]

        # Print the result
        print('TEST: BATCH %04d/%04d | LOSS: %.4f | ACC %.4f' %
              (batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))

## Checking

# obtain one batch of test images
dataiter = iter(loader)
images, labels = dataiter.next()

images_flatten = images.view(images.size(0), -1)
# get sample outputs
output = net(images_flatten)
# prep images for display
images = images.numpy()

# output is resized into a batch of images
output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

