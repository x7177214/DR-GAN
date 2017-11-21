import tarfile
from six.moves import urllib
import sys
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random
import skimage.io as io
import skimage.transform as st

import torch
import torch.utils.data as Data

if __name__ == '__main__':

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data_files = os.listdir('/home/hank/Desktop/123')
            print(self.data_files)
            # sort(self.data_files)

        def __getitem__(self, idx): # All data MUST have the same size before return
            return st.resize(io.imread('/home/hank/Desktop/123/'+self.data_files[idx]), [20, 20])

        def __len__(self):
            return len(self.data_files)

    dset = MyDataset()
    loader = Data.DataLoader(dataset=dset, batch_size=2, num_workers=2, shuffle=True)

    for x in loader: #iterate dataset
        print(x.shape)
        print('gg')

    