import torch
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

from grad_cam.grad_cam import GradCam
from models.simple import SimpleNetwork
from models.resnet_based import ResnetBased


# model = ResnetBased()
model = SimpleNetwork()

# model.load_state_dict(torch.load("saved_models/resnetbased_unmasked_1epochs.pt"))
model.load_state_dict(torch.load("saved_models/simple_uint8_masked_3epochs.pt"))

# wavy artifacts but the range is [0, 255]
# image = io.imread("../project-data/xtion1_float/depth/pant/01_01_0070.png", dtype='uint8')
# image = io.imread("../project-data/xtion1_float/depth/shirt/01_01_0020.png", dtype='uint8')
# image = io.imread("../project-data/xtion1_float/depth/sweater/01_02_0040.png", dtype='uint8')
image = io.imread("../project-data/xtion1_float/depth/tshirt/01_01_0050.png", dtype='uint8')
image = resize(image, (240, 320))

# no wavy artifacts, but the intensity range is off
# image = io.imread("../project-data/xtion1_uint8/depth/shirt/01_03_0047.png")
# avgpool = model.features()
# fc = model.classifier()
grad_cam = GradCam(model=model, target_layer_names=["0"], use_cuda=False)

image = torch.tensor(image).float()
image.unsqueeze_(0)
image.unsqueeze_(0)
target_index = None

mask = grad_cam(image, target_index)
plt.imshow(mask)
plt.colorbar()
plt.show()

# Show prediction 'animation'
# for i in range(36, 51):
#     image = io.imread("../project-data/xtion1_float/depth/pant/01_01_00%d.png" % i, dtype='uint8')
#     image = resize(image, (240, 320))
#     image = torch.tensor(image).float()
#     image.unsqueeze_(0)
#     image.unsqueeze_(0)
#     grad_cam(image, None)