import torch.nn as nn
import torch
from typing import List, Optional

class ShortcutProjection(nn.Module):
    """
    This class is for the use of dimensions modifications of the identity mapping (x)
    We use the number of channels and stride used by the residual block to match the dimensions using a 1x1 kernel convolution.
    """
    def __init__(self,in_channels:int,out_channels:int,stride:int):
        
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self,x:torch.Tensor):
        return self.bn(self.conv(x))
    
class ResidualBlock(nn.Module):
    """
    A stack of 2 3x3 conv layers with a shortcut connection
    """
    def __init__(self,in_channels:int,out_channels:int,stride:int):
        super().__init__()
        #enters 64 channels of 56x56 , to keep h*w dim we pad 1s 
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1) 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels=in_channels,out_channels=out_channels,stride=stride)
        else:
            self.shortcut = nn.Identity() #when channels are the same, just copy x, no padding required.
        
        self.act2 = nn.ReLU()
        
    def forward(self,x:torch.Tensor):
        shortcut = self.shortcut(x)
    
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)   
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out+= shortcut
        
        out = self.act2(out)
        
        return out
        
class ResNetBase(nn.Module):
    def __init__(self, n_blocks: List[int], n_channels: List[int],
                #  bottlenecks: Optional[List[int]] = None,
                 img_channels: int = 3, first_kernel_size: int = 7):
        super().__init__()
        
        assert len(n_blocks) == len(n_channels)
        # assert bottlenecks is not None or len(bottlenecks) == len(n_channels)
    
        self.conv = nn.Conv2d(in_channels=img_channels,out_channels=n_channels[0],
                              kernel_size=first_kernel_size,stride=2,padding=first_kernel_size//2)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        blocks = []
        prev_channels = n_channels[0]
        
        for i,channels in enumerate(n_channels):
            stride = 2 if len(blocks) == 0 else 1 
            blocks.append(ResidualBlock(in_channels=prev_channels,out_channels=channels,stride=stride))
            prev_channels = channels
            
            #rest of residual blocks without change in size thus stride = 1, take away one since you dont need the initial,
            for _ in range(n_blocks[i]-1):
                blocks.append(ResidualBlock(in_channels=channels,out_channels=channels,stride=1))
        
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self,x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = self.blocks(x)
        
        return x