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
from tqdm import tqdm

inpsiz = 784
hidensiz = 500
numclases = 10
numepchs = 100
bachsiz = 32
l_r = 0.001

E = 6

data = np.load('data.npy')
test_data = np.load('test_data.npy')
label = np.load('label.npy')
test_label = np.load('test_label.npy')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Average:
    def __init__(self):
        self.val = 0
        self.count = 0
    
    def record(self, new_val):
        if self.count == 0:
            self.val = new_val
        else:
            self.val = new_val / self.count + self.val * (1 - 1 / self.count)
        
        self.count += 1.0
    
    def get_val(self):
        return self.val

def lower_upper(delta_x: torch.Tensor, w_: torch.Tensor):
    global upper_avg
    kernel_size, out_ch = w_.shape
    uppers = [0 for _ in range(out_ch)]
    for k in range(out_ch):
        w_k = w_[:, k]
        rank = torch.argsort(torch.relu(w_k), descending=True)
        sign_val = w_k[rank[:E]] * delta_x[rank[:E]]
        neg_loc = (sign_val <= 0)
        temp = torch.sqrt(torch.square(torch.norm(w_k)) - torch.square(torch.norm(w_k[rank[:E][neg_loc]])))
        uppers[k] = temp
    return torch.tensor(uppers)

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

        self.use_uv = use_uv
    
    def forward(self, input):
        return myconv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, self.use_uv, self)


def scale_to_01(x):
    return (x - x.min()) * 0.999999 / (x.max() - x.min())

upper_avg = Average()

def lower_upper(delta_x: torch.Tensor, w_: torch.Tensor):
    kernel_size, out_ch = w_.shape
    uppers = [0 for _ in range(out_ch)]
    for k in range(out_ch):
        w_k = w_[:, k]
        rank = torch.argsort(w_k, descending=True)
        sign_val = w_k[rank[:E]] * delta_x[rank[:E]]
        neg_loc = (sign_val <= 0)
        uppers[k] = torch.square(torch.norm(w_k)) - torch.square(torch.norm(w_k[rank[0:E][neg_loc]]))
    return torch.tensor(uppers)
    

def myconv2d(conv_inp, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, use_uv=True, conv=None):
    """
    Function to process an input with a standard convolution
    """
    # print('@@@')
    hash_conv = HashUtils.get_hash_func(conv)
    batch_size, in_channels, in_h, in_w = conv_inp.shape
    out_channels, in_channels, kh, kw = conv.weight.shape
    out_h = int((in_h - kh + 2 * conv.padding[0]) / conv.stride[0] + 1)
    out_w = int((in_w - kw + 2 * conv.padding[1]) / conv.stride[1] + 1)
    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=conv.dilation, padding=conv.padding, stride=conv.stride)
    bias = conv.bias if conv.bias is not None else 0

    inp_unf = unfold(conv_inp).transpose(1, 2)
    w_ = conv.weight.view(conv.weight.size(0), -1).t()  # (9, 32)
    out_unf = inp_unf.matmul(w_)
    out_unf_true = out_unf + bias
    sparsity = (torch.count_nonzero(out_unf_true <= 0) / torch.numel(out_unf_true)).item()

    hash_out = hash_conv(conv_inp)

    hash_out = scale_to_01(hash_out)

    hash_out = hash_out * inp_unf.shape[1] * 0.4
    bin_id_map = (hash_out).int().reshape(-1)  # get bin id

    bin_count = bin_id_map.bincount()  # num of pixels in each bin
    ref_bin_ids = (bin_count > 0).nonzero()
    ref_bin_ids = ref_bin_ids.squeeze(1)
    ref_bin_id_to_idx = {}

    w_norm = torch.norm(w_, dim=0)  # (len=channel num)
    count_hash = 0.0
    count_locality = 0.0
    count_try_skip = 0.0  # num of pixels that try to skip
    total = float(batch_size * out_w * out_h * out_channels)
    inp_unf, out_unf = inp_unf.reshape(-1, inp_unf.size(-1)), out_unf.reshape(-1, out_unf.size(-1))
    for batch_i in tqdm(range(batch_size)):
        for ij in range(out_w * out_h):
            idx = batch_i * out_w * out_h + ij
            x_ij = inp_unf[idx, :]
        
            bin_id = bin_id_map[idx].item()
            if bin_id in ref_bin_id_to_idx:
                ref_idx = ref_bin_id_to_idx[bin_id]
                if idx != ref_idx:
                    delta_x_ij = x_ij - inp_unf[ref_idx, :]
                    delta_x_ij_norm = torch.norm(delta_x_ij)  # ||x_ij - x_ref||
                    w_norm_lower = lower_upper(delta_x_ij, w_)
                    ref_mul = out_unf[ref_idx, :] + delta_x_ij_norm * w_norm_lower + bias  # (len=channel num)
                    # ref_mul = out_unf[ref_idx, :] + delta_x_ij_norm * w_norm + bias  # (len=channel num)
                    count_hash += torch.count_nonzero(ref_mul <= 0).item()
                    global upper_avg
                    upper_avg.record(torch.mean(w_norm_lower / w_norm).detach().numpy())
                    # print(upper_avg.get_val())
        
            else:
                ref_bin_id_to_idx[bin_id] = idx
    
    print('{:.3f}'.format(count_hash/total))
    print(len(ref_bin_id_to_idx) / len(bin_id_map))

    out = out_unf_true.transpose(1, 2)
    out = out.view(batch_size, out_channels, out_h, out_w)  # (1, 32, 26, 26)
    return out.float()


class HashUtils(object):
    def __init__(self):
        self.tag = 'HashUtils'
    
    @staticmethod
    def get_hash_func(conv: MyConv2d):
        """FIXME profiling weight output """
        
        tmp = nn.Conv2d(in_channels=conv.in_channels, out_channels=1, kernel_size=conv.kernel_size, stride=conv.stride,
                       padding=conv.padding)
        avg_w = torch.mean(conv.weight, dim=0, keepdim=True)
        avg_b = torch.mean(conv.bias, dim=0, keepdim=True)
        tmp.weight.data = avg_w.data
        tmp.bias.data = avg_b.data
        return tmp


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
test_trainldr = torch.utils.data.DataLoader(dataset=test_ds,shuffle=True,
                                            batch_size=10)

# print('train data len is ', len(trainldr.dataset))
print('test data len is ', len(test_trainldr.dataset))


class neural_network(nn.Module):
    def __init__(self):
        super(neural_network, self).__init__()
        self.conv1 = MyConv2d(1, 32, 3, 1, use_uv=False)
        self.conv2 = MyConv2d(32, 64, 3, 1, use_uv=False)
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
modl.load_state_dict(torch.load('cnn.pt', map_location=device), strict=False)
    
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
    print('Acc = {}'.format(correct / total))

# torch.save(modl.state_dict(), 'cnn.pt')
