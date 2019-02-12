import os
import re
import shutil
from shutil import copyfile
import cv2

import numpy as np
from skimage import io
from matplotlib import pyplot as plt

# this script can be used for both masking out (removing) the background
# and masking out the cloth (signal) itself

SRC = "../../project-data/pirstines/Views_handparsed"
DST = "../../project-data/continous_masked_out"

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

for garmet in ['pant', 'shirt', 'sweater', 'towel', 'tshirt']:
# for garmet in ['pant']:
    make_folders(garmet)
    moves = os.listdir('%s/%s' % (SRC, garmet))
    n_moves = len(moves)
    print("garmet %s has %d moves" % (garmet, n_moves))

    move_number = 0
    for move in moves:
        move_number += 1
    # for gi in range(1, 3+1):
    #     for m in range(1, 10+1):
        move_dir = '%s/%s/%s' % (SRC, garmet, move)
        # # some move directories are hand-removed if they were found noisy (not anymore)
        # try:
        #     os.stat(move_dir)
        # except:
        #     continue
        files = os.listdir(move_dir)

        # make a `move` destination folder for both depth and masked
        dst_depth_move_path = '%s/depth/%s/%d' % (DST, garmet, move_number)
        dst_masked_move_path = '%s/masked/%s/%d' % (DST, garmet, move_number)
        try: os.stat(dst_depth_move_path)
        except: os.makedirs(dst_depth_move_path)
        try: os.stat(dst_masked_move_path)
        except: os.makedirs(dst_masked_move_path)

        # move depth frames
        for f in files:
            # skip `mask` folder located in each `move` directory
            if f == 'mask':
                continue
            depth_filepath = os.path.join(move_dir, f)
            i = get_number(f)
            if i == None:
                print("problem with %s %d, skipping" % (move_dir, i))
                continue

            to_filename_format = "{0:}_{1:04}.png".format(move, int(i))
            # to_depth_str = os.path.join(DST, 'depth', garmet, to_filename_format)
            to_depth_str = os.path.join(dst_depth_move_path, to_filename_format)
            copyfile(depth_filepath, to_depth_str)

            mask_filename = 'imagemask%s.png' % i
            mask_filepath = os.path.join(move_dir, 'mask', mask_filename)

            # depth = io.imread(depth_filepath, dtype='float64')
            depth = cv2.imread(depth_filepath, cv2.IMREAD_GRAYSCALE)
            # depth = io.imread(depth_filepath) // 256
            # mask = io.imread(mask_filepath, dtype='uint8') // 255
            mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)

            # Invert mask
            # mask = mask - 255           
            masked = depth * mask
            
            # plt.imshow(masked)
            # plt.colorbar()
            # plt.show()

            # to_filepath = os.path.join(DST, 'masked', garmet, to_filename_format)
            to_filepath = os.path.join(dst_masked_move_path, to_filename_format)
            io.imsave(to_filepath, masked)
