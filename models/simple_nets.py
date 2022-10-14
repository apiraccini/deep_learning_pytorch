# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_classes = 10

# simple neural nets

class VanillaNet(nn.Module):
    
    '''
    Combines convolutions and pooling to learn spatial features from images.
    This way, we achieve local operation on neighborhoods, translation 
    invariance, fewer parameters in the model.
    '''

    def __init__(self, n_chans1=16):
        super().__init__()
        self.n_chans1 = n_chans1
        
        # convolution + pooling
        self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=n_chans1, kernel_size=3, padding=1))
        self.add_module('conv2', nn.Conv2d(in_channels=n_chans1, out_channels=n_chans1//2, kernel_size=3, padding=1))
        
        # linear
        self.add_module('fc1', nn.Linear(n_chans1//2 * 8 * 8, 32))
        self.add_module('fc2', nn.Linear(32, n_classes))

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = out.view(-1, self.n_chans1//2 * 8 * 8)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class DropoutNet(nn.Module):

    '''
    Uses Dropout (2014), a procedure which amounts to zero out a proportion of 
    outputs after activation functions during training.
    Avoids overfitting and is somewhat similar to regularization/weight decay
    or data augmentation. Dropout2d zeroes 2D channels.
    '''

    def __init__(self, n_chans1=16):
        super().__init__()
        self.n_chans1 = n_chans1
        
        # convolution + pooling
        self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=n_chans1, kernel_size=3, padding=1))
        self.add_module('conv2', nn.Conv2d(in_channels=n_chans1, out_channels=n_chans1//2, kernel_size=3, padding=1))
        
        # linear
        self.add_module('fc1', nn.Linear(n_chans1//2 * 8 * 8, 32))
        self.add_module('fc2', nn.Linear(32, n_classes))

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = F.dropout(F.max_pool2d(out, 2), p=0.4)
        out = torch.relu(self.conv2(out))
        out = F.dropout(F.max_pool2d(out, 2), p=0.4)
        out = out.view(-1, self.n_chans1//2 * 8 * 8)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class BatchnormNet(nn.Module):

    '''
    Uses Batch normalization (2015), which standardizes inputs using running 
    batch mean and std during training and uses whole data estimates during
    inference (inputs still need to be standardized beforehand).
    Acts as a regularizer (so is an alternative to dropout) and allows for
    greater learning rates
    '''

    def __init__(self, n_chans1=16):
        super().__init__()
        self.n_chans1 = n_chans1
        
        # convolution + batchnorm + pooling
        self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=n_chans1, kernel_size=3, padding=1))
        self.add_module('bn1', nn.BatchNorm2d(num_features=n_chans1))
        self.add_module('conv2', nn.Conv2d(in_channels=n_chans1, out_channels=n_chans1 // 2, kernel_size=3, padding=1))
        self.add_module('bn2', nn.BatchNorm2d(num_features=n_chans1 // 2))

        # linear
        self.add_module('fc1', nn.Linear(n_chans1//2 * 8 * 8, 32))
        self.add_module('fc2', nn.Linear(32, n_classes))

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.max_pool2d(torch.relu(out), 2)
        out = self.bn2(self.conv2(out))
        out = F.max_pool2d(torch.relu(out), 2)
        out = out.view(-1, self.n_chans1//2 * 8 * 8)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    print('These classes implement some simple neural networks.')