import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# images from xtion1
# I consider this normalisation to be good (but it is NOT due to the uint8 cast)
depth = io.imread('../project-data/Views_handparsed/pant/pant1_move1/imagedepth80.png')
mask = io.imread('../project-data/Views_handparsed/pant/pant1_move1/mask/imagemask80.png') / 256
depth = depth*mask
mmax = np.max(depth)
# mmin = np.min(depth)
mmin = list(set(np.sort(depth.ravel())))[1]
print(mmin)
print(mmax)
# min-max normalisation
depth = (depth-mmin) / (mmax-mmin)
# depth = depth.clip(0)
plt.imshow(depth)
plt.colorbar()
plt.show()

# imaes from kinect
# I consider this normalisation to be bad
# depth = io.imread('../project-data/kinect_sorted_pirstine/pant/12/clothes_12_capture_1_depth.png')
# mask = io.imread('../project-data/kinect_sorted_pirstine/pant/12/clothes_12_capture_1_mask.png', dtype='uint8') // 2
# depth = depth*mask
# # print(depth[200])
# # mmax = np.max(depth)
# # depth = depth / mmax
# plt.imshow(depth)
# plt.colorbar()
# plt.show()