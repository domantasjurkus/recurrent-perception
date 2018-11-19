import torch
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

from grad_cam.grad_cam import GradCam
from models.simple import SimpleNetwork

model = SimpleNetwork()
model.load_state_dict(torch.load("saved_models/simple_uint8_masked_3epochs.pt"))

# wavy artifacts but the range is [0, 255]
image = io.imread("../project-data/xtion1_uint8/depth/shirt/01_03_0047.png", dtype='uint8') # 480,640
image = resize(image, (240, 320))
print(image.shape)

# no wavy artifacts, but the intensity range is off
# image = io.imread("../project-data/xtion1_uint8/depth/shirt/01_03_0047.png")

grad_cam = GradCam(model=model, target_layer_names=["features"], use_cuda=False)

image = torch.tensor(image).float()
image.unsqueeze_(0)
image.unsqueeze_(0)
target_index = 1

mask = grad_cam(image, target_index)
plt.imshow(mask)
plt.colorbar()
plt.show()