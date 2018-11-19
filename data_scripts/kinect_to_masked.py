import os
import re
import shutil
from shutil import copyfile
import numpy as np
# import scipy.misc
from matplotlib import pyplot as plt

from skimage import io

SRC = '../../project-data/kinect_sorted_pirstine'
DEST = '../../project-data/kinect_masked'

def make_folders(garmet):
    try:
        os.stat(DEST)
    except:
        os.makedirs(DEST)

    try:
        os.stat('%s/%s' % (DEST,garmet))
    except:
        os.makedirs('%s/%s' % (DEST,garmet))

    try:
        os.stat('%s/%s/depth' % (DEST,garmet))
    except:
        os.makedirs('%s/%s/depth' % (DEST,garmet))

    try:
        os.stat('%s/%s/masked' % (DEST,garmet))
    except:
        os.makedirs('%s/%s/masked' % (DEST,garmet))

def filename_contains(filename, key):
    match = re.search(r'%s' % key, filename)
    return match != None

def remobe_black(img2d, n_bins=256):
    limit = 1.0/n_bins
    return list(filter(lambda pix: pix > limit, img2d.ravel()))

for garmet in ['pant', 'shirt', 'sweater', 'tshirt']:
# for garmet in ['pant']:
    make_folders(garmet)
    garmet_dir = os.listdir('%s/%s' % (SRC,garmet))
    n_instances = len(garmet_dir)
    print("garmet %s has %d instances" % (garmet, n_instances))

    for num in garmet_dir:
        path = os.path.join(SRC, garmet, num)
        files = os.listdir(path)

        depth_files = list(filter(lambda f: filename_contains(f, 'depth'), files))
        mask_files = list(filter(lambda f: filename_contains(f, 'mask'), files))
        
        dict = {} # {'13': {'depth': <filepath>, 'mask': <filepath>}}
        for df in depth_files:
            i = df.split('_')[3]
            dict[i] = {}
            dict[i]['depth'] = os.path.join(path, df)

        for mf in mask_files:
            i = mf.split('_')[3]
            dict[i]['mask'] = os.path.join(path, mf)

        for gi in dict.keys():
            gi = '9'
            depth = io.imread(dict[gi]['depth'])
            mask = io.imread(dict[gi]['mask']) // 2

            # some image resolutions are inconsistent (yay)
            try:
                depth = depth * mask
                # depth = np.uint8(depth * mask)
                pass
            except:
                print('warning: garmet %s gi %s has odd resolution' % (garmet, gi))
            
            mmax = np.max(depth)
            depth = depth / mmax
            
            print(depth[150])
            plt.imshow(depth)
            plt.colorbar()
            plt.show()

            # save one depth frame per class, then bother about processing everything
            try: os.stat('../../project-data/kinect_subset/%s' % garmet)
            except: os.mkdir('../../project-data/kinect_masked_subset/%s' % garmet)
            scipy.misc.imsave('../../project-data/kinect_masked_subset/%s/%s.png' % (garmet, garmet), depth)

            # depth = remobe_black(depth, n_bins=256)
            
            # plt.hist(depth.ravel(), bins=256, range=(0.0, 1.0))
            # plt.show()

            break
        break

        #     try:
        #         from_str = dict[gi]['depth']
        #         to_str = 'kinect/{0}/depth/{1:02}.png'.format(garmet, int(gi))
        #         copyfile(from_str, to_str)

        #         from_str = dict[gi]['mask']
        #         to_str = 'kinect/{0}/mask/{1:02}.png'.format(garmet, int(gi))
        #         copyfile(from_str, to_str)
        #         print(from_str)
        #         print(to_str)
        #     except:
        #         print('error: cound not copy garmet %s instance %s' % (garmet, gi))
