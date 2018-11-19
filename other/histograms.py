# attempts to normalise histograms
# def filter_out(img2d):
#     limit = 1.0/128
#     return list(filter(lambda pix: pix > limit, img2d))

# train_itr = iter(train_loader)
# train_batch = train_itr.next()
# train_image = train_batch[0][0][0].detach().numpy()
# kinect_image = iter(kinect_loader).next()[0][0][0] .detach().numpy()
# kinect_image = kinect_image.astype('uint8')
# print(train_image[100])

# This thing is giving me a headache, will use numpy implementation
# cv2.equalizeHist(kinect_image)

# clahe = cv2.createCLAHE()
# cl1 = clahe.apply(kinect_image)
# show_image(train_image)

# from util import normalise_histogram
# normalise_histogram(train_image)
# plt.show()

# train_filtered = filter_out(train_image.ravel())
# kinect_filtered = filter_out(kinect_image.ravel())
# print(len(kinect_filtered))

# plt.hist(train_image.ravel(), bins=256, range=(0.0, 1.0))
# plt.show()
# plt.hist(kinect_image.ravel(), bins=128, range=(0.0, 1.0))
# plt.show()
