# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_classes = 10

# simple neural nets

class ConvBlock(nn.Module):

    '''Convolution block'''

    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1,):
        super(ConvBlock, self).__init__()
        
        # convolution + batchnorm + dropout
        self.add_module('conv', nn.Conv2d(in_chans, out_chans,
                                          kernel_size=kernel_size, padding=padding, bias=False))
        self.add_module('bn', nn.BatchNorm2d(num_features=out_chans))

        # parameter initialization
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.bn.weight, 0.5)
        torch.nn.init.zeros_(self.bn.bias)
    
    def forward(self, x):
        out = F.dropout2d(torch.relu(self.bn(self.conv(x))), p=0.1)
        return out

class ConvNet(nn.Module):

    '''Combines convolutions, pooling, batch normalization and dropout'''

    def __init__(self, n_chans=16, n_blocks=2):
        super().__init__()

        self.n_chans = n_chans
        
        # first convolution
        self.add_module('conv1', ConvBlock(3, out_chans=n_chans))
        self.add_module('bn1', nn.BatchNorm2d(num_features=n_chans))

        # intermediate convolution blocks
        self.add_module('convblocks',
                        nn.Sequential(*(n_blocks * [ConvBlock(in_chans=n_chans, out_chans=n_chans)])))

        # linear
        self.add_module('fc1', nn.Linear(n_chans * 8 * 8, 32))
        self.add_module('fc2', nn.Linear(32, n_classes))

    def forward(self, x):

        out = torch.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        out = F.max_pool2d(self.convblocks(out), 2)
        out = out.view(-1, self.n_chans * 8 * 8)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    print('''\n
        This class implement a convolutional neural network using a user made
        convolution + batchnorm + dropout block.
        ''')