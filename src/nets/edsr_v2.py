import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += residual
        
        return out

class Net(nn.Module):
    def __init__(self, num_res_blocks=32, num_channels=256, upscale_factor=4):
        super(Net, self).__init__()
        
        self.num_res_blocks = num_res_blocks
        
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(num_channels))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        self.upscale = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * upscale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        
        self.conv3 = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        
        residual = out
        
        out = self.res_blocks(out)
        
        out = self.conv2(out)
        out += residual
        
        out = self.upscale(out)
        
        out = self.conv3(out)
        
        return out
