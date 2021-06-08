from concurrent import futures
import os
import glob
import torch.nn as nn
import torch.nn.functional as F

import threading

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageCms
import time
import random
import os
import numpy as np

import pandas as pd

import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, x):
        return x + torch.randn(x.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
class NinetiesRotation:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

class FolderSet(Dataset):
  def __init__(this, root_dir, resolution = (256, 256), center=False):
    this.resolution = resolution
    if center:
      this.crop_transform = transforms.CenterCrop(resolution)
    else:
      this.crop_transform = transforms.Compose([
                                              transforms.RandomCrop(resolution, padding=None, pad_if_needed=True),
                                              transforms.RandomHorizontalFlip(),
                                              NinetiesRotation()
                                              ])
    # Transforms a image to a grayscale image
    this.X_transforms = transforms.Compose([
                                            transforms.Grayscale(num_output_channels=1),
                                            #AddGaussianNoise(mean=0.,std=1.)
                                            ])
    this.toTensor = transforms.Compose([transforms.ToTensor()])
    
    this.files = glob.glob(f"{root_dir}/*.png")
    random.shuffle(this.files)
    this.length = len(this.files)

      
  def __len__(this):
    return this.length
    
  def __getitem__(this,idx):
    im = Image.open(this.files[idx])
    im = im if im.mode == "RGB" else im.convert("RGB")
    
    transformed_im = this.crop_transform(im)
    
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    LAB = ImageCms.applyTransform(transformed_im, rgb2lab)
    L, A, B = LAB.split()
    
    Xs = this.toTensor(L)
    Ys = torch.cat((this.toTensor(A), this.toTensor(B)))
    
    return (Xs, Ys)