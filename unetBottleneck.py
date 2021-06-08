import torch.nn as nn
import torch.nn.functional as F
import os
import torch

class CnnBlock(nn.Module):
  def __init__(self, in_channels, out_channels, skip_final_activation=False):
    super().__init__()
    self.skip_final_activation = skip_final_activation
    self.activation = F.relu
    intermediate = max(3, in_channels//4)
    self.conv1 = nn.Conv2d(in_channels, intermediate, 1, stride=1, padding=0)
    self.conv2 = nn.Conv2d(intermediate, intermediate, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(intermediate, out_channels, 1, stride=1, padding=0)
    self.skip = nn.Conv2d(in_channels, out_channels, 1)
  
  def forward(self, x):
    input_x = x
    x = self.activation(self.conv1(x))
    x = self.activation(self.conv2(x))
    x = self.conv3(x)
    x = x + self.skip(input_x)
    if not self.skip_final_activation:
      x = self.activation(x)
    return x

class StackedBlocks(nn.Module):
  def __init__(self, in_channels, out_channels, n_blocks):
    super().__init__()
    self.n_blocks = n_blocks
    if n_blocks == 0: # one Conv2d
      self.blocks = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1), nn.ReLU())
    elif n_blocks == 1: # two Conv2d with a skip connection
      self.blocks = CnnBlock(in_channels, out_channels)
    elif n_blocks > 1:
      self.blocks = nn.Sequential(CnnBlock(in_channels, out_channels), *[CnnBlock(out_channels, out_channels) for i in range(n_blocks-1)])
      self.skip = nn.Conv2d(in_channels, out_channels, 1)
    else:
      raise ValueError("n_blocks must be larger than 0, it was:", n_blocks)

  def forward(self, x):
    input_x = x
    x = self.blocks(x)
    return self.skip(input_x) + x if self.n_blocks > 1 else x

class Encoder(nn.Module):
  def __init__(self, nchannels, nblocks):
    super().__init__()
    self.nchannels = nchannels
    self.pool = nn.MaxPool2d(2)
    self.blocks = nn.ModuleList([StackedBlocks(nchannels[i], nchannels[i+1], nblocks) for i in range(len(nchannels)-2)])
    self.blocks.append(CnnBlock(nchannels[-2], nchannels[-1]))

  def forward(self, x):
    features = []
    for block in self.blocks:
      x = block(x)
      features.append(x)
      x = self.pool(x)
    return features

class UpscaleBlock(nn.Module): # A*A*C -> 2A*2A*C/2
  def __init__(self, in_channels, out_channels):
    super().__init__()
    #self.upscaleLayer = nn.PixelShuffle(2) # A*A*C -> 2A*2A*C/4
    self.upscaleLayer = nn.Upsample(scale_factor=2, mode='nearest')
    self.pad = nn.ReflectionPad2d(1)
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=0, stride=1)
    self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    x = self.pad(self.upscaleLayer(x))
    x = self.conv1(x)
    x = self.activation(x)
    return x

class Decoder(nn.Module):
  def __init__(self, nchannels):
    super().__init__()
    self.nchannels = nchannels
    self.upconvs = nn.ModuleList([UpscaleBlock(nchannels[i], nchannels[i]//2) for i in range(len(nchannels)-1)])
    self.blocks = nn.ModuleList([CnnBlock(nchannels[i], nchannels[i+1]) for i in range(len(nchannels)-1)])
    self.finalBlocks = nn.Sequential(nn.Conv2d(nchannels[-1], nchannels[-1], 3, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(nchannels[-1], 2, 3, padding=1))

  def forward(self, x, encoder_features):
    for i in range(len(self.nchannels)-1):
      x = self.upconvs[i](x)
      x = torch.cat([x, encoder_features[i]], dim=1)
      x = self.blocks[i](x)
      temp = encoder_features[i]
      del temp
      encoder_features[i] = None
    x = self.finalBlocks(x)
    return x

class UNet(nn.Module):
  # Important! The side lengths of the input image must be divisible depth times by 2. Add padding to nearest multiple when evaluating
  # Safe size: current_size + current_size % 2**(len(nchannels)-1) 
  # Pad to safe size, then crop to correct upscaled size afterwards
  def __init__(self, depth=5, init_channels=64, nblocks=1):
    super().__init__()
    #nchannels=[64,128,256,512]
    self.nchannels = [init_channels * 2**i for i in range(depth)]
    self.encoder = Encoder([1] + self.nchannels, nblocks)
    self.decoder = Decoder(self.nchannels[::-1]) # reverse

  def forward(self, x):
    encoder_features = self.encoder(x)
    out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
    return out
