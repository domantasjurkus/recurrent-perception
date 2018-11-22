import os
import re
import shutil
from shutil import copyfile

SRC = "../../project-data/pirstine/Views_handparsed"
DST = "../../project-data/xtion1"

def make_folders(garmet):
    try:
        os.stat(DST)
    except:
        os.makedirs(DST)

    try:
        os.stat('%s/%s' % (DST, garmet))
    except:
        os.makedirs('%s/%s' % (DST, garmet))

    try:
        os.stat('%s/%s/depth' % (DST, garmet))
    except:
        os.makedirs('%s/%s/depth' % (DST, garmet))

    try:
        os.stat('%s/%s/mask' % (DST, garmet))
    except:
        os.makedirs('%s/%s/mask' % (DST, garmet))

def get_minmax_indices(move_dir):
    mmin = 999
    mmax = 0
    filenames = [f for f in os.listdir(move_dir) if os.path.isfile(os.path.join(move_dir, f))]
    for f in filenames:
        match = re.search(r'\d+\d*', f)
        a = int(match[0])
        mmin = a if a < mmin else mmin
        mmax = a if a > mmax else mmax

    print(mmin, mmax)
    return mmin, mmax

def is_mask_file(filename):
    if filename == 'mask':
        return False
    match = re.search(r'mask', filename)
    return match != None

def move_masks_in_views(garmet, gi, m):
    move_folder = 'Views/%s/%s%d_move%d' % (garmet, garmet, gi, m)

    # delete ./mask folder if present
    # try:
    #     shutil.rmtree('%s/mask' % move_folder)
    # except:
    #     print('could not delete %s/mask for some reason' % move_folder)
    #     pass

    try:
        all_files = os.listdir(move_folder)
    except:
        print(move_folder, "missing, skipping")
        return

    mask_files = list(filter(is_mask_file, all_files))
    # mask_filepaths = [os.path.join(move_folder, f) for f in mask_files]
    print(mask_files)

    # make ./mask folder
    try:
        os.makedirs('%s/mask' % move_folder)
    except:
        print("garmet %s %d move %s already has ./mask, skipping" % (garmet, gi, m))

    for f in mask_files:
        from_str = os.path.join(move_folder, f)
        to_str = os.path.join(move_folder, 'mask', f)
        try:
            shutil.move(from_str, to_str)
            pass
        except:
            print(to_str, 'already exists')
            pass

for garmet in ['pant', 'shirt', 'sweater', 'tshirt']:
    make_folders(garmet)
    n_moves = len(os.listdir('Views/%s' % garmet))
    print("garmet %s has %d moves" % (garmet, n_moves))
    for gi in range(1, 3+1):
        for m in range(1, 10+1):
            
            move_masks_in_views(garmet, gi, m)
            
            try:
                mmin, mmax = get_minmax_indices('Views/%s/%s%d_move%d' % (garmet, garmet, gi, m))
            except Exception as e:
                print('skipping %s %d move %d' % (garmet, gi, m))
                continue

            for i in range(mmin, mmax+1):
                pants = 'Views/%s/%s%d_move%d/imagedepth%d.png' % (garmet, garmet, gi, m, i)
                mask = 'Views/%s/%s%d_move%d/mask/imagemask%d.png' % (garmet, garmet, gi, m, i)

                try:
                    from_str = 'Views/%s/%s%d_move%d/imagedepth%d.png' % (garmet, garmet, gi, m, i)
                    to_str = 'single_folder/{0}/depth/{1:02}_{2:02}_{3:04}.png'.format(garmet, gi, m, i)
                    copyfile(from_str, to_str)

                    from_str = 'Views/%s/%s%d_move%d/mask/imagemask%d.png' % (garmet, garmet, gi, m, i)
                    to_str = 'single_folder/{0}/mask/{1:02}_{2:02}_{3:04}.png'.format(garmet, gi, m, i)
                    copyfile(from_str, to_str)
                except Exception as e:
                    print('skip %s %d move %d' % (garmet, gi, m))
                    print(e)
                    break
