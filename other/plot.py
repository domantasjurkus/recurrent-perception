import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread("../project-data/xtion1_float/depth/pant/01_01_0050.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../project-data/xtion1_float/depth/pant/01_01_0060.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("../project-data/xtion1_float/depth/pant/01_01_0075.png", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread("../project-data/xtion1_float/depth/pant/01_01_0085.png", cv2.IMREAD_GRAYSCALE)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
plt.set_cmap('gray')
ax1.imshow(img1)
ax2.imshow(img2)
ax3.imshow(img3)
ax4.imshow(img4)

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')

plt.show()