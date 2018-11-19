import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def show(img):
    npimg = img.numpy()
    trans = np.transpose(npimg, (1,2,0))
    plt.imshow(trans, interpolation='nearest')
    plt.show()

img = torch.randn((1, 768, 1024))
print(img.size())

imglist = [img, img]
g = vutils.make_grid(imglist, padding=100)
print(g.shape)

show(g)