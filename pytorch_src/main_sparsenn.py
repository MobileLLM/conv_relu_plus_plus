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


inpsiz = 784
hidensiz = 500
numclases = 10
numepchs = 100
bachsiz = 32
l_r = 0.001

data = np.load('data.npy')
test_data = np.load('test_data.npy')
label = np.load('label.npy')
test_label = np.load('test_label.npy')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        u, s, v = torch.svd(self.weight.view(self.weight.size(0), -1).t(), some=True, compute_uv=True)
        self.u = nn.Parameter(u)
        self.s = nn.Parameter(s)
        self.v = nn.Parameter(v)
        self.use_uv = use_uv
    
    def reset_usv(self):
        u, s, v = torch.svd(self.weight.view(self.weight.size(0), -1).t(), some=True, compute_uv=True)
        self.u = nn.Parameter(u)
        self.s = nn.Parameter(s)
        self.v = nn.Parameter(v)
    
    def forward(self, input):
        return myconv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, self.u, self.s, self.v, self.use_uv)



def myconv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, u=None, s=None, v=None, use_uv=True):
    """
    Function to process an input with a standard convolution
    """
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)
    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inp_unf = unfold(input)
    w_ = weight.view(weight.size(0), -1).t()    # w_(9, 32)
    
    X = inp_unf.transpose(1, 2)
    out_unf = (X @ w_ + bias)
    
    r = len(s)//2
    overhead = r / out_channels + r / len(s)
    # print('overhead:', overhead)
    # print(r)
    # upper = (X @ (torch.mm(torch.mm(u, torch.diag(s)), v.t())) + bias)     # u: (9,9),  v.t(): (9,32)
    upper = (X @ (torch.mm(torch.mm(u[:, :r], torch.diag(s[:r])), v.t()[:r, :])) + bias)
    
    # acc = torch.count_nonzero(upper == upper2)/torch.numel(upper2)
    
    mask = (upper <= 0)
    
    MFLOPS = 2 * out_channels * out_w * out_h * in_channels * kh * kw / 1e6
    
    reduced = (torch.count_nonzero(mask) / torch.numel(mask)).item()
    result_flops = 1 - reduced + overhead
    # print('sparseNN: {:.2f}'.format(result_flops) , '= 1 -', '{:.2f}'.format(reduced), '+', '{:.2f}'.format(overhead))
    
    print(result_flops * MFLOPS, MFLOPS)
    
    # w_norm = torch.norm(w_, dim=0)  # (len=channel num)
    # count_locality = 0.0
    # total = float(batch_size * out_w * out_h * out_channels)
    #
    # for batch_i in range(batch_size):
    #     for ij in range(out_w * out_h):
    #         # idx = batch_i * out_w * out_h + ij
    #         x_ij = X[batch_i, ij, :]  # inp_unf: (1000, 676, 9)
    #
    #         if ij > 0:
    #             delta_x_ij_norm = torch.norm(x_ij - X[batch_i, ij-1, :])  # ||x_ij - x_ref||
    #             prev_upper = out_unf[batch_i, ij-1, :]
    #             norm_norm = delta_x_ij_norm * w_norm
    #             ref_mul = prev_upper + norm_norm
    #             count_locality += torch.count_nonzero(ref_mul <= 0).item()
    # skip_ratio_ = count_locality / total
    #
    # # print('skip_ratio {:.3f}'.format())
    # our_result = 1 - skip_ratio_ + 1 / out_channels
    # print('Our: {:.2f}'.format(our_result), '= 1 -', '{:.2f}'.format(skip_ratio_), '+', '{:.2f}'.format( 1 / out_channels))
    # print('sparseNN {:.3f}'.format((torch.count_nonzero(mask)/torch.numel(mask)).item()))

    if use_uv:
        out_unf[mask] = 0

    out_unf = out_unf.transpose(1, 2)
    out = out_unf.view(batch_size, out_channels, out_h, out_w)  # (1, 32, 26, 26)
    return out.float()


nn.Conv2d = MyConv2d


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
test_trainldr = torch.utils.data.DataLoader(dataset=test_ds, shuffle=True,
                                            batch_size=1000)

print('train data len is ', len(trainldr.dataset))
print('test data len is ', len(test_trainldr.dataset))


class neural_network(nn.Module):
    def __init__(self):
        super(neural_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, use_uv=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, use_uv=True)
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
        return x

modl = neural_network().to(device)
modl.load_state_dict(torch.load('cnn.pt', map_location=device), strict=True)

modl.conv1.reset_usv()
modl.conv2.reset_usv()

criter = nn.CrossEntropyLoss().to(device)
optim = torch.optim.Adam(modl.parameters(), lr=l_r)
nttlstps = len(trainldr)
for epoch in range(numepchs):
    modl.train()
    for x, (imgs, lbls) in enumerate(trainldr):
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        outp = modl(imgs)

        losses = criter(outp, lbls)

        optim.zero_grad()
        losses.backward()
        optim.step()
    
modl.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for x, (imgs, lbls) in enumerate(test_trainldr):
        imgs = imgs.reshape(-1, 1, 28, 28).to(device)
        lbls = lbls.to(device)
        outp = modl(imgs)
        pred = outp.data.max(1, keepdim=True)[1].to(device)
        correct += pred.eq(lbls.data.view_as(pred)).sum().detach().cpu().item()
        total += torch.numel(pred)
        break
    print('Acc = {}'.format(correct / total))

torch.save(modl.state_dict(), 'cnn.pt')