import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class AdverserialModel(nn.Module):
  def __init__(this, high_res):
    super().__init__()
    this.model = nn.Sequential(
      nn.Conv2d(2, 16, 3,padding=1), # 3*3*3*16 = 432
      nn.BatchNorm2d(16),
      nn.LeakyReLU(0.2, inplace=True), # 256
      nn.Conv2d(16, 32, 3,padding=1,stride=2), # 2
      nn.BatchNorm2d(32),
      nn.LeakyReLU(0.2, inplace=True), # 256
      nn.Conv2d(32, 64, 3,padding=1), # 18 432
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.2, inplace=True), # 128
      nn.Conv2d(64, 128, 3,padding=1, stride=2), # 73 728
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True), # 64
      nn.Conv2d(128, 256, 3,padding=1, stride=2), # 
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True), # 32
      nn.Conv2d(256, 512, 3,padding=1, stride=2), # 
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, inplace=True), # 16
      nn.Conv2d(512, 1024, 3,padding=1, stride=2), #
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.2, inplace=True), # 8
      nn.Conv2d(1024, 2048, 3,padding=1, stride=2), #
      nn.BatchNorm2d(2048),
      nn.LeakyReLU(0.2, inplace=True), # 4
      nn.AdaptiveAvgPool2d(2),
      
      nn.Flatten(),
      
      nn.Linear(2048 * 2**2, 128), 
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(128, 1)
    )

  def forward(this, x):
    return this.model(x)

def psnr(real, fake):
  return -10*torch.log10(F.mse_loss(real, fake))

# Copyright (c) 2013 Anders Hast
# Uppsala University
# http://www.cb.uu.se/~aht
# 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind.
#

def kernel(spline): 
  if spline == 'Cubic':
      #B  = [-1,1,-1,1;0,0,0,1;1,1,1,1;8,4,2,1];
      #M  = inv(B);
      M = torch.tensor([[-1,3,-3,1], [3,-6,3,0], [-2,-3,6,-1], [0,6,0,0]])*1/6
      u  = torch.tensor([[0.125], [0.25], [0.5], [1]])
      up = torch.tensor([0.75,1,1,0]).view(-1, 1)
      upp= torch.tensor([3,2,0,0]).view(-1, 1)
  elif spline == 'Catmull-Rom': #
      M = torch.tensor([[-1,3,-3,1], [2,-5,4,-1], [-1,0,1,0], [0,2,0,0]])*0.5
      u  = torch.tensor([0.125, 0.25, 0.5, 1]).view(-1, 1)
      up = torch.tensor([0.75,1,1,0]).view(-1, 1)
      upp= torch.tensor([3, 2, 0, 0]).view(-1, 1)
  elif spline == 'Trigonometric':
      M  = [[1,1,0,1], [1,torch.sqrt(3/4),0.5,0.5], [1,0.5,torch.sqrt(3/4),-0.5], [1,0,1,-1]]
      M  = torch.inverse(M)
      u  = [1,torch.sqrt(1/2),torch.sqrt(1/2),0].view(-1, 1)
      up = [0,-torch.sqrt(1/2),torch.sqrt(1/2),-2].view(-1, 1)
      upp= [0,-torch.sqrt(1/2),-torch.sqrt(1/2),0].view(-1, 1)
  else:
      raise ValueError('Spline unknown!')
  """ elif spline == 'Bezier':
      M=[1,0,0,0;-3,3,0,0;3, -6,3,0; -1,3,-3,1]';
      u  = [0.125;0.25;0.5;1];
      up = [0.75;1;1;0];
      upp= [3;2;0;0];
  elif spline == 'B-Spline':
      M=[-1,3,-3,1;3,-6,3,0;-3,0,3,0;1,4,1,0]*1/6;
      u  = [0.125;0.25;0.5;1];
      up = [0.75;1;1;0];
      upp= [3;2;0;0]; """
  
  k  = torch.mm(u.T, M)
  d  = torch.mm(up.T, M)
  d2  = torch.mm(upp.T, M)
  return (k, d, d2)

def superHast(y, device):
  # Trigonometric
  dk = torch.tensor([-0.006127921758831, 0.196582449765983, -1.328234947353380, -0.000000000000001, 1.328234947353381, -0.196582449765985, 0.006127921758831]).view(1, -1)
  kk = torch.tensor([0.004333095030250, -0.074492438854197, 0.245666904969751, 0.648984877708396, 0.245666904969750, -0.074492438854198, 0.004333095030250]).view(1, -1)
  a = torch.matmul(dk.T, kk).view(1,1,7,7).expand(3,-1,-1,-1).float().to(device)
  b = torch.matmul(kk.T, dk).view(1,1,7,7).expand(3,-1,-1,-1).float().to(device)
  Hx = F.conv2d(y, a, groups=y.shape[1])
  Hy = F.conv2d(y, b, groups=y.shape[1])
  return (Hx**2 + Hy**2 + 1e-12).sqrt()

def catmullHast(y, device):
  # Catmull-Rom
  dk = torch.tensor([-0.0078125, 0.15625, -0.7890625, 0, 0.7890625, -0.15625, 0.0078125]).view(1, -1)
  kk = torch.tensor([0.00390625, -0.0703125, 0.24609375, 0.640625, 0.24609375, -0.0703125, 0.00390625]).view(1, -1)
  a = torch.matmul(dk.T, kk).view(1,1,7,7).expand(3,-1,-1,-1).float().to(device)
  b = torch.matmul(kk.T, dk).view(1,1,7,7).expand(3,-1,-1,-1).float().to(device)
  Hx = F.conv2d(y, a, groups=y.shape[1])
  Hy = F.conv2d(y, b, groups=y.shape[1])
  return (Hx**2 + Hy**2 + 1e-12).sqrt()
