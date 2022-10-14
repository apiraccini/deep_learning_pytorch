# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_classes = 10

# simple neural nets

class ResBlock(nn.Module):

    '''ResNet intermediate block with skip connection'''

    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        
        # convolution + batchnorm
        self.add_module('conv', nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(num_features=n_chans))

        # parameter initialization
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.bn.weight, 0.5)
        torch.nn.init.zeros_(self.bn.bias)
    
    def forward(self, x):
        out = torch.relu(self.bn(self.conv(x)))
        return out + x

class DeepResNet(nn.Module):
    
    '''
    Residual nets (2015): deep net with multiple skip connections, which 
    alleviates the vanishing gradient problem allowing for deeper nets.
    Skip connection: out = layer(x) + x
    The structure is similar to torchvision pretrained ResNet50 etc. 
    '''
    
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        
        # first convolution
        self.add_module('conv1', nn.Conv2d(3, n_chans1, kernel_size=3, padding=1))
        self.add_module('bn1', nn.BatchNorm2d(num_features=n_chans1))
        
        # resnet blocks with skip connection
        self.add_module('resblocks', nn.Sequential(*(n_blocks * [ResBlock(n_chans=n_chans1)])))

        # linear
        self.add_module('fc1', nn.Linear(8 * 8 * n_chans1, 32))
        self.add_module('fc2', nn.Linear(32, n_classes))

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.max_pool2d(torch.relu(out), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    print('''\n
        This class implement a deep convolutional neural network using 
        multiple skip connection blocks.
        ''')