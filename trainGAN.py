import csv

import argparse
from concurrent import futures
import os
import re
import sys

import boto3
import botocore
import tqdm

import torch.nn as nn
import torch.nn.functional as F

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageCms
import time
from multiprocessing import Process, Queue
import os
import numpy as np

import pandas as pd

import random

from dataset import *
from hqset import *
from net import *
from unet import *
from test import predict

from collections import namedtuple

import torch
from torchvision import models
from torchvision.io.image import read_image, ImageReadMode

import common_parameters
from losses import *

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
  torch.multiprocessing.freeze_support()
    
  print('cuda' if torch.cuda.is_available() else 'cpu')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  net = UNet(depth=4).to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=common_parameters.learning_rate)
  
  disc = AdverserialModel(64).to(device)

  optimizer_disc = torch.optim.Adam(disc.parameters(), lr=common_parameters.learning_rate)

  if len(sys.argv) != 3: raise RuntimeError("Two command-line arguments must be given, the model's filename and the type of loss")
  filename = sys.argv[1]
  loss_str = sys.argv[2]
  # criterion is a function that takes the arguments (real_imgs, fake_imgs) in that order!
  if loss_str == "mse":
    criterion = F.mse_loss
  elif loss_str == "l1":
    criterion = F.l1_loss
  else:
    raise RuntimeError(loss_str + " is not a valid loss")

  writer = SummaryWriter(common_parameters.relative_path + 'runs/' + filename.split('.')[0])
  filename = common_parameters.relative_path + filename


  iterations, train_losses, val_losses = loadNetGAN(filename, net, optimizer, disc, optimizer_disc, device)
  best_loss = min(val_losses) if len(val_losses) > 0 else 1e6
  print("Best validation loss:", best_loss)
  iteration = iterations[-1] if len(iterations) > 0 else -1
  

  net.train()
  net.to(device)
  

  batch_size = common_parameters.batch_size

  traindata = FolderSet(common_parameters.relative_path + "train", resolution=(64,64))
  validdata = FolderSet(common_parameters.relative_path + "valid", resolution=(64,64), center = True)

  dataset = DataLoader(traindata, batch_size=batch_size, shuffle=True)
  validation_dataset = DataLoader(validdata, batch_size=batch_size)
  
  validation_data = [i for i in validation_dataset]
  validation_size = len(validation_data)
  
  #dataset = DataLoader(FolderSet("text"), batch_size=10, num_workers = 7)
  
  print("Datasets loaded")
  print_every = 50
  save_every = 200
  i = iteration
  
  srgb_p = ImageCms.createProfile("sRGB")
  lab_p  = ImageCms.createProfile("LAB")

  rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
  lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")
 
 
  speed_mini_im = Image.open("speed-mini.png")
  speed_mini_im = speed_mini_im if speed_mini_im.mode == "RGB" else speed_mini_im.convert("RGB")
  
  speed_mini = transforms.ToTensor()(ImageCms.applyTransform(speed_mini_im, rgb2lab).split()[0]).to(device).float() # L
    
  for epoch in range(1000):  # loop over the dataset multiple times

      running_lossD, running_lossG, running_loss = [],[],[]
      train_loss = 0.0
      for data in dataset:
          i += 1
          if i > common_parameters.end_iterations - 1:
            break
          # get the inputs; data is a list of [inputs, labels]
          inputs, real = data
          inputs = inputs.to(device)
          real = real.to(device)
          
          batch_size = len(inputs)
          
          #real_labels = torch.ones(batch_size).unsqueeze(-1).to(device)
          
          net.zero_grad()
          
          real_out = disc(real)
          fakes = net(inputs)
          
          fake_out = disc(fakes)
          errG = (torch.mean((real_out - torch.mean(fake_out) + 1)**2) + torch.mean((fake_out - torch.mean(real_out) - 1)**2))/2
          
          loss = errG + 0.5*criterion(real, fakes)
          loss.backward(retain_graph=True)
          optimizer.step()

          disc.zero_grad()
          fake_out = disc(fakes.detach())
          errD = (torch.mean((real_out - torch.mean(fake_out) - 1)**2) + torch.mean((fake_out - torch.mean(real_out) + 1)**2))/2
          
          errD.backward()
          running_lossD.append(errD.item())
          
          optimizer_disc.step()


          running_lossG.append(errG.item())
          loss_item = loss.item()
          running_loss.append(loss_item)
          train_loss += loss_item


          # print statistics
          if i % print_every == 0:
            print('[%d, %5d] loss: %.4f' %
                  (epoch, i, sum(running_loss)/len(running_loss)))
            print('[%d, %5d] lossG: %.4f' %
                  (epoch, i, sum(running_lossG)/len(running_lossG)))
            print('[%d, %5d] lossD: %.4f' %
                  (epoch, i, sum(running_lossD)/len(running_lossD)))
            writer.add_scalar("train/loss", sum(running_loss)/len(running_loss), i)
            writer.add_scalar("train/loss_generator", sum(running_lossG)/len(running_lossG), i)
            writer.add_scalar("train/loss_discriminator", sum(running_lossD)/len(running_lossD), i)
            net.train()
            running_lossD, running_lossG, running_loss = [],[],[]
          if i % save_every == 0:
            train_losses.append(train_loss/save_every)
            iterations.append(i)
            train_loss = 0.0
            saveNetGAN(filename, net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
            with torch.no_grad():
              net.eval()
              criterion_loss = 0.0
              psnr_score = 0
              psnrs = []
              for inputs, labels in validation_data:
                inputs = inputs.to(device)
                real_val = labels.to(device)
                fakes_val = net(inputs)
                criterion_loss += criterion(real_val, fakes_val).item()
                psnr_score += psnr(real_val, fakes_val).item()
                
              criterion_loss /= validation_size
              psnr_score /= validation_size
              validation_loss = criterion_loss
              val_losses.append(validation_loss)
              writer.add_scalar("valid/loss", validation_loss, i)
              writer.add_scalar("valid/PSNR", psnr_score, i)
              
              speed_mini_predict_AB = net(speed_mini.unsqueeze(0)).squeeze()
              speed_mini_predict_LAB = transforms.ToPILImage()(torch.cat((speed_mini, speed_mini_predict_AB)))
              speed_mini_predict = ImageCms.applyTransform(speed_mini_predict_LAB, lab2rgb)
              
              
              writer.add_image("validation image", transforms.ToTensor()(speed_mini_predict), i)
              writer.add_image("validation image A", speed_mini_predict_AB[0,...].unsqueeze(0), i)
              writer.add_image("validation image B", speed_mini_predict_AB[1,...].unsqueeze(0), i)
              
              
              print("Validation loss:", validation_loss, "Mean PSNR:", psnr_score)
              net.train()
              if validation_loss < best_loss:
                saveNetGAN(filename + "_best", net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
                print(f"New best loss: {best_loss} -> {validation_loss}")
                best_loss = validation_loss
            print("Saved model!")
      # This code makes sure that we break both loops if the inner loop is broken out of:
      else:
        continue
      break
  writer.close() 
              
                
