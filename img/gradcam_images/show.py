import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img0 = mpimg.imread('depth_1.jpg')
img1 = mpimg.imread('depth_2.jpg')
img2 = mpimg.imread('depth_3.jpg')
img3 = mpimg.imread('depth_4.jpg')

img4 = mpimg.imread('masked_1.jpg')     
img5 = mpimg.imread('masked_2.jpg')
img6 = mpimg.imread('masked_3.jpg')
img7 = mpimg.imread('masked_4.jpg')

f, axarr = plt.subplots(2,4)
axarr[0,0].imshow(img0, cmap='gray')
axarr[0,1].imshow(img1, cmap='gray')
axarr[0,2].imshow(img2, cmap='gray')
axarr[0,3].imshow(img3, cmap='gray')

axarr[1,0].imshow(img4, cmap='gray')
axarr[1,1].imshow(img5, cmap='gray')
axarr[1,2].imshow(img6, cmap='gray')
axarr[1,3].imshow(img7, cmap='gray')

for axrow in axarr:
    for ax in axrow:
        ax.axis('off')

plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)

plt.show()