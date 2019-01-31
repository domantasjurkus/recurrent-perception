import os
import re
import shutil
from shutil import copyfile, copytree, rmtree
import cv2

import numpy as np
from skimage import io
from matplotlib import pyplot as plt

SRC = "../../project-data/pirstines/Views_handparsed"
DST = "../../project-data/xtion1_moves"

# 
# Not finished
# 

def make_folders(garmet):
    try: os.stat(DST)
    except: os.makedirs(DST)

    # depth
    try: os.stat('%s/depth' % DST)
    except: os.makedirs('%s/depth' % DST)
    try: os.stat('%s/depth/%s' % (DST, garmet))
    except: os.makedirs('%s/depth/%s' % (DST, garmet))

    # masked
    try: os.stat('%s/masked' % DST)
    except: os.makedirs('%s/masked' % DST)
    try: os.stat('%s/masked/%s' % (DST, garmet))
    except: os.makedirs('%s/masked/%s' % (DST, garmet))


def is_mask_file(filename):
    if filename == 'mask':
        return False
    match = re.search(r'mask', filename)
    return match != None

def get_number(filename):
    m = re.search(r'\d+', filename)
    return m[0] if m != None else m

for garmet in ['pant', 'shirt', 'sweater', 'tshirt']:
# for garmet in ['pant']:
    make_folders(garmet)
    n_moves = len(os.listdir('%s/%s' % (SRC, garmet)))
    print("garmet %s has %d moves" % (garmet, n_moves))

    for gi in range(1, 3+1):
        for m in range(1, 10+1):
            garment_move = '%s%d_move%d' % (garmet, gi, m)
            move_from_dir = '%s/%s/%s' % (SRC, garmet, garment_move)
            move_depth_to_dir = '%s/%s/%s/%s' % (DST, 'depth', garmet, garment_move)
            move_masked_to_dir = '%s/%s/%s/%s' % (DST, 'masked', garmet, garment_move)
            
            # some move directories are hand-removed if they were found noisy
            try:
                os.stat(move_from_dir)
            except:
                print('dir %s removed, skipping' % move_from_dir)
                continue
            files = os.listdir(move_from_dir)

            # copy depth frames
            # delete old dir first
            try:
                rmtree(move_depth_to_dir)
            except:
                pass
            copytree(move_from_dir, move_depth_to_dir)
            rmtree(os.path.join(move_depth_to_dir, 'mask'))

            # make masked dir
            try:
                rmtree(move_masked_to_dir)
            except:
                pass
            os.makedirs(move_masked_to_dir)

            # make masked frames
            for f in files:
                if f == 'mask':
                    continue
                depth_filepath = os.path.join(move_from_dir, f)
                i = get_number(f)
                if i == None:
                    print("problem with %s %d, skipping" % (move_from_dir, i))
                    continue

                # to_filename_format = "{0:02}_{1:02}_{2:04}.png".format(gi, m, int(i))
                # to_depth_str = os.path.join(DST, 'depth', garmet, to_filename_format)
                # copyfile(depth_filepath, to_depth_str)

                # mask image
                mask_filename = 'imagemask%s.png' % i
                mask_filepath = os.path.join(move_from_dir, 'mask', mask_filename)

                depth = cv2.imread(depth_filepath, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                masked = depth * mask

                # plt.imshow(masked)
                # plt.colorbar()
                # plt.show()

                # to_filepath = os.path.join(move_masked_to_dir, to_filename_format)
                to_filepath = os.path.join(move_masked_to_dir, mask_filename)
                io.imsave(to_filepath, masked)