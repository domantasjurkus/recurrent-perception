import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img0 = mpimg.imread('../../project-data/continuous_depth/pant/1/pant1_move1_0050.png')
img1 = mpimg.imread('../../project-data/continuous_depth/pant/1/pant1_move1_0060.png')
img2 = mpimg.imread('../../project-data/continuous_depth/pant/1/pant1_move1_0070.png')
img3 = mpimg.imread('../../project-data/continuous_depth/pant/1/pant1_move1_0075.png')
img4 = mpimg.imread('../../project-data/continuous_depth/pant/1/pant1_move1_0080.png')
img5 = mpimg.imread('../../project-data/continuous_depth/pant/1/pant1_move1_0090.png')

img6 = mpimg.imread('../../project-data/continuous_masked/pant/1/pant1_move1_0050.png')
img7 = mpimg.imread('../../project-data/continuous_masked/pant/1/pant1_move1_0060.png')
img8 = mpimg.imread('../../project-data/continuous_masked/pant/1/pant1_move1_0070.png')
img9 = mpimg.imread('../../project-data/continuous_masked/pant/1/pant1_move1_0075.png')
img10 = mpimg.imread('../../project-data/continuous_masked/pant/1/pant1_move1_0080.png')
img11 = mpimg.imread('../../project-data/continuous_masked/pant/1/pant1_move1_0090.png')


f, axarr = plt.subplots(2,6)
axarr[0,0].imshow(img0, cmap='gray')
axarr[0,1].imshow(img1, cmap='gray')
axarr[0,2].imshow(img2, cmap='gray')
axarr[0,3].imshow(img3, cmap='gray')
axarr[0,4].imshow(img4, cmap='gray')
axarr[0,5].imshow(img5, cmap='gray')

axarr[1,0].imshow(img6, cmap='gray')
axarr[1,1].imshow(img7, cmap='gray')
axarr[1,2].imshow(img8, cmap='gray')
axarr[1,3].imshow(img9, cmap='gray')
axarr[1,4].imshow(img10, cmap='gray')
axarr[1,5].imshow(img11, cmap='gray')

for axrow in axarr:
    for ax in axrow:
        ax.axis('off')

plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)

plt.show()