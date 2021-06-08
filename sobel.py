import matplotlib.pyplot as plt
from dataset import *
from losses import *
import losses
def sobel_filter(y, device):
  kernel_x = torch.tensor([[1, 0, -1],[2,0,-2],[1,0,-1]]).view(1,1,3,3).float().to(device)
  kernel_y = torch.tensor([[1, 2, 1],[0,0,0],[-1,-2,-1]]).view(1,1,3,3).float().to(device)
  Gx = F.conv2d(y, kernel_x, groups=y.shape[1])
  Gy = F.conv2d(y, kernel_y, groups=y.shape[1])
  return (Gx**2 + Gy**2 + 1e-8).sqrt()
 
if __name__ == '__main__':
  device = torch.device("cpu")

  imf = input("Enter file:")
  
  x = Image.open(imf).convert("RGB")
  plt.imshow(x)
  plt.show(block=False)
  
  
  y = transforms.ToTensor()(x)
  #y_sobel = sobel_filter(y.mean(0).unsqueeze(0).unsqueeze(0),device).squeeze().unsqueeze(0).expand(3,-1,-1)
  
  y_sobel = losses.sobel_filter(y.unsqueeze(0),device).squeeze()
  y_sobel_color = losses.superHast(y.unsqueeze(0),device).squeeze()*4
 
  plt.figure()
  plt.imshow(transforms.ToPILImage()(y_sobel))
  plt.show(block=False)
  plt.figure()
  plt.imshow(transforms.ToPILImage()(y_sobel_color))
  plt.show()
  