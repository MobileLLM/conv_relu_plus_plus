import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as trans
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from torch.nn.modules.utils import _single, _pair, _triple

numepchs = 100
bachsiz = 32
l_r = 0.001

data = np.load('data.npy')
test_data = np.load('test_data.npy')
label = np.load('label.npy')
test_label = np.load('test_label.npy')


class MyConv2d(torch.nn.modules.conv._ConvNd):
    """
    Implements a standard convolution layer that can be used as a regular module
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', use_uv=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
    
    def forward(self, input):
        return myconv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def myconv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Function to process an input with a standard convolution
    """
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)
    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    
    N = out_channels
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gama = torch.tensor([1.0 - i/N for i in range(N)]).to(device)

    inp_unf = unfold(input)
    w_ = weight.view(weight.size(0), -1).t()  # w_(9, 32)
    
    X = inp_unf.transpose(1, 2)
    out_unf = (X @ w_ + bias)*gama
    
    out_unf = out_unf.transpose(1, 2)
    out = out_unf.view(batch_size, out_channels, out_h, out_w)  # (1, 32, 26, 26)

    out[:, int(out_channels*0.8):, :, :] = 0
    return out.float()

# nn.Conv2d = MyConv2d

class MyDS(Dataset):
    """Face Landmarks dataset."""
    
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        
        if self.transform:
            return self.transform(torch.tensor(x.reshape(1, 28, 28), dtype=torch.float32)), torch.tensor(y,
                                                                                                         dtype=torch.long)
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


t = transforms.Compose([
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
ds = MyDS(data, label, t)
test_ds = MyDS(test_data, test_label, t)

trainldr = torch.utils.data.DataLoader(dataset=ds,
                                       batch_size=bachsiz,
                                       shuffle=True)
test_trainldr = torch.utils.data.DataLoader(dataset=test_ds,
                                            batch_size=1000)

print('train data len is ', len(trainldr.dataset))
print('test data len is ', len(test_trainldr.dataset))


class neural_network(nn.Module):
    # def __init__(self, inpsiz, hidensiz, numclases):
    #      super(neural_network, self).__init__()
    #      self.inputsiz = inpsiz
    #      self.l1 = nn.Linear(inpsiz, hidensiz)
    #      self.relu = nn.ReLU()
    #      self.l2 = nn.Linear(hidensiz, numclases)
    # def forward(self, y):
    #      outp = self.l1(y)
    #      outp = self.relu(outp)
    #      outp = self.l2(outp)
    #      return outp
    def __init__(self):
        super(neural_network, self).__init__()
        self.conv1 = MyConv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

modl = neural_network().to(device)
modl.load_state_dict(torch.load('cnn_IDP.pt', map_location=device), strict=True)
# modl = neural_network(inpsiz, hidensiz, numclases).to(device

criter = nn.CrossEntropyLoss().to(device)
optim = torch.optim.Adam(modl.parameters(), lr=l_r)
nttlstps = len(trainldr)
# for epoch in range(numepchs):
#     modl.train()
#     for x, (imgs, lbls) in enumerate(trainldr):
#         imgs = imgs.to(device)
#         lbls = lbls.to(device)
#
#         outp = modl(imgs)
#
#         losses = criter(outp, lbls)
#
#         optim.zero_grad()
#         losses.backward()
#         optim.step()
#
#     modl.eval()
with torch.no_grad():
    modl.eval()
    correct = 0
    for x, (imgs, lbls) in enumerate(test_trainldr):
        imgs = imgs.reshape(-1, 1, 28, 28).to(device)
        lbls = lbls.to(device)
        outp = modl(imgs)
        pred = outp.data.max(1, keepdim=True)[1].to(device)
        correct += pred.eq(lbls.data.view_as(pred)).sum().detach().cpu().item()
        losses = criter(outp, lbls)

    print('Acc = {}'.format(correct / len(test_trainldr.dataset)))

# torch.save(modl.state_dict(), 'cnn_IDP.pt')