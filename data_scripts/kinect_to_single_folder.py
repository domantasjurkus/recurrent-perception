import os
import re
import shutil
from shutil import copyfile

def make_folders(garmet):
    try:
        os.stat('kinect')
    except:
        os.makedirs('kinect')

    try:
        os.stat('kinect/%s' % garmet)
    except:
        os.makedirs('kinect/%s' % garmet)

    try:
        os.stat('kinect/%s/depth' % garmet)
    except:
        os.makedirs('kinect/%s/depth' % garmet)

    try:
        os.stat('kinect/%s/mask' % garmet)
    except:
        os.makedirs('kinect/%s/mask' % garmet)

def filename_contains(filename, key):
    match = re.search(r'%s' % key, filename)
    return match != None

ROOT = 'kinect_sorted'

for garmet in ['pant', 'shirt', 'sweater', 'tshirt']:
# for garmet in ['pant']:
    make_folders(garmet)
    garmet_dir = os.listdir('%s/%s' % (ROOT,garmet))
    n_instances = len(garmet_dir)
    print("garmet %s has %d instances" % (garmet, n_instances))

    for num in garmet_dir:
        path = os.path.join(ROOT, garmet, num)
        files = os.listdir(path)
        # print(files)

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
            try:
                from_str = dict[gi]['depth']
                to_str = 'kinect/{0}/depth/{1:02}.png'.format(garmet, int(gi))
                copyfile(from_str, to_str)
                # print(from_str)
                # print(to_str)

                from_str = dict[gi]['mask']
                to_str = 'kinect/{0}/mask/{1:02}.png'.format(garmet, int(gi))
                copyfile(from_str, to_str)
                print(from_str)
                print(to_str)
            except:
                print('error: cound not copy garmet %s instance %s' % (garmet, gi))
