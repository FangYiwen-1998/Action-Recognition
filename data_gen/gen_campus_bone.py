import os
import numpy as np
from numpy.lib.format import open_memmap

paris = {
    'ntu/xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),

    'kinetics': ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                 (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)),
    'campus': ((0, 0), (1, 2), (2, 4), (3, 1), (3, 4), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), 
          (5, 6), (7, 5), (8, 6), (9, 7), (10, 8), (7, 8), (9, 10), (13, 14), (15, 16), (7, 11), (8, 12),
          (11, 5), (12, 6), (11, 12), (13, 11), (14, 12), (15, 13), (16, 14), (9, 15), (10, 16))
}

sets = {
    'train', 'val'
}

# 'ntu/xview', 'ntu/xsub',  'kinetics'
datasets = {
    'ntu/xview', 'ntu/xsub',
}
# bone
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Campus Data Converter.')
    parser.add_argument(
        '--inter', default=1, type=int)
    arg = parser.parse_args()
    if arg.inter:
        pth1 = "./data/campus/{}_data_joint.npy"
        pth2 = './data/campus/{}_data_bone.npy'
    else:
        pth1 = "./data/campus/{}_data_joint2.npy"
        pth2 = './data/campus/{}_data_bone2.npy'
    for se in sets:
        data = np.load(pth1.format(se))
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            pth2.format(se),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))
        fp_sp[:, :C, :, :, :] = data
        for v1, v2 in tqdm(paris['campus']):
             fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

#for dataset in datasets:
#    for set in sets:
#        print(dataset, set)
#        data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
#        N, C, T, V, M = data.shape
#        fp_sp = open_memmap(
#            '../data/{}/{}_data_bone.npy'.format(dataset, set),
#            dtype='float32',
#            mode='w+',
#            shape=(N, 3, T, V, M))
#
#        fp_sp[:, :C, :, :, :] = data
#        for v1, v2 in tqdm(paris[dataset]):
#            if dataset != 'kinetics':
#                v1 -= 1
#                v2 -= 1
#            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
