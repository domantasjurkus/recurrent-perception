import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from grad_cam.grad_cam import GradCam
from models.cifar_based import CifarBased
from models.simple import SimpleNetwork
from models.resnet_based import ResnetBased

model = CifarBased()
# model = SimpleNetwork()
# model = ResnetBased()

model.load_state_dict(torch.load("saved_models/cifarbased_unmasked_50epochs.pt"))
# model.load_state_dict(torch.load("saved_models/simple_uint8_masked_3epochs.pt"))
# model.load_state_dict(torch.load("saved_models/resnetbased_unmasked_1epochs.pt"))

# wavy artifacts but the range is [0, 255]
# image = io.imread("../project-data/xtion1_float/depth/pant/01_01_0070.png", dtype='uint8')
# image = io.imread("../project-data/xtion1_float/depth/shirt/01_01_0020.png", dtype='uint8')
# image = io.imread("../project-data/xtion1_float/depth/sweater/01_02_0040.png", dtype='uint8')
image = cv2.imread("../project-data/xtion1_float/depth/tshirt/01_01_0050.png", cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (240, 320))

grad_cam = GradCam(model=model, target_layer_names=["10"], use_cuda=False)

image = torch.tensor(image).float()
image.unsqueeze_(0)
image.unsqueeze_(0)
target_index = None

mask = grad_cam(image, target_index)

# I am not sure what I'm showing so I better keep it off the proposal
# def show_cam_on_image(img, mask):
#     print(img.shape)
#     print(mask.shape)
#     heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     print(img[50][50])
#     print(mask[50][50])
#     # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
#     plt.imshow(cam)
#     plt.colorbar()
#     plt.show()

# mask = cv2.resize(mask, (640, 480))
# image = image[0].detach().numpy()
# image = np.transpose(image, (1,2,0)) / 255
# show_cam_on_image(image, mask)

# Show prediction 'animation'
# for i in range(40, 51):
#     image = cv2.imread("../project-data/xtion1_float/depth/tshirt/01_01_00%d.png" % i, cv2.IMREAD_GRAYSCALE)
#     image = torch.tensor(image).float()
#     image.unsqueeze_(0)
#     image.unsqueeze_(0)
#     grad_cam(image, None)