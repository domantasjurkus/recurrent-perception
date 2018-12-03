import os
import re
import shutil
from shutil import copyfile
import cv2

import numpy as np
from skimage import io
from matplotlib import pyplot as plt

SRC = "../../project-data/pirstines/Views_handparsed"
DST = "../../project-data/xtion1_masked_out"

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
            move_dir = '%s/%s/%s%d_move%d' % (SRC, garmet, garmet, gi, m)
            
            # some move directories are hand-removed if they were found noisy
            try:
                os.stat(move_dir)
            except:
                continue
            files = os.listdir(move_dir)

            # move depth frames
            for f in files:
                if f == 'mask':
                    continue
                depth_filepath = os.path.join(move_dir, f)
                i = get_number(f)
                if i == None:
                    print("problem with %s %d, skipping" % (move_dir, i))
                    continue

                to_filename_format = "{0:02}_{1:02}_{2:04}.png".format(gi, m, int(i))
                to_depth_str = os.path.join(DST, 'depth', garmet, to_filename_format)
                copyfile(depth_filepath, to_depth_str)

                mask_filename = 'imagemask%s.png' % i
                mask_filepath = os.path.join(move_dir, 'mask', mask_filename)

                # depth = io.imread(depth_filepath, dtype='float64')
                depth = cv2.imread(depth_filepath, cv2.IMREAD_GRAYSCALE)
                # depth = io.imread(depth_filepath) // 256
                # mask = io.imread(mask_filepath, dtype='uint8') // 255
                mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)

                # Invert mask
                mask = mask - 255

                # This uint8 cast is giving me ridges, but it's probably a hack
                # masked = np.uint8(depth * mask)
                
                masked = depth * mask
                
                # plt.imshow(masked)
                # plt.colorbar()
                # plt.show()

                to_filepath = os.path.join(DST, 'masked', garmet, to_filename_format)
                io.imsave(to_filepath, masked)