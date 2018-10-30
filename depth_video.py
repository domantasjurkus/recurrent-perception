import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader import GarmetDataset

WIDTH = 320
HEIGHT = 240

def get_training_data_iterator():
    folder = GarmetDataset(root='../project-data/single_folder/training', masked=True)
    return iter(DataLoader(folder, batch_size=1, num_workers=1, shuffle=False))

def save_image(tensor, i):
    x = tensor.view(-1, HEIGHT, WIDTH)
    plt.imshow(x[0, ...])
    # plt.colorbar()
    plt.savefig('animation/%d' % i,bbox_inches='tight',transparent=True, pad_inches=0)
    # plt.show()

# def save_image(tensor):
#     image = tensor.view(-1, HEIGHT, WIDTH)
#     fig = plt.figure(frameon=False)
#     fig.set_size_inches(WIDTH, HEIGHT)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(image, aspect='normal')
#     fig.savefig('hamster.png')

itr = get_training_data_iterator()

# skip all pants
for i in range(0, 1446):
    itr.next()

for i in range(12, 36):
    image = itr.next()
    save_image(image[0], i)
    
