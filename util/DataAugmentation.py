#!/usr/bin/env python
# encoding: utf-8

# Data Augmentation class which is used with DataLoader
# Assume numpy array face images with B x C x H x W  [-1~1]

import scipy as sp
import numpy as np
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
import pdb

class FaceIdPoseDataset(Dataset):

    #  assume images  as B x C x H x W  numpy array
    def __init__(self, images, IDs, poses, transform=None):

        self.images = images
        self.IDs = IDs
        self.poses = poses
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        ID = self.IDs[idx]
        pose = self.poses[idx]
        if self.transform:
            image = self.transform(image)

        return [image, ID, pose]

class FaceIdPoseDataset2(Dataset):

    # assume
    #             images: N x C x H x W  numpy array (N: # of total training data)
    #             images_paths: N x "str" list of image path
    #             IDs: N x num_ID numpy array
    #             illus: N x num_illumination numpy array

    def __init__(self, images_paths, IDs, illus, transform=None, random_flip=True):
        self.images_paths = images_paths
        self.IDs = IDs
        self.illus = illus
        self.transform = transform
        self.random_flip = random_flip

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        im_p = self.images_paths[idx]
        ID = self.IDs[idx]
        illu = self.illus[idx]

        # read image
        image = io.imread(im_p)
        
        '''
        image = FaceCrop(image)
        image = Resize(image)
        '''

        # horizontal flip the image with 50% chance
        if self.random_flip:
            if int(np.random.randint(2)) == 0:
                image = np.fliplr(image)

        #[0,255] -> [-1,1]
        image = image / 255.0
        image = 2.0 * image - 1.0

        # RGB -> BGR
        image = image[:,:,[2,1,0]]

        # H x W x C -> C x H x W
        image = image.transpose(2, 0, 1)

        # RandomCrop
        if self.transform:
            image = self.transform(image)
 
        return [image, ID, illu]

# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         self.data_files = os.listdir('/home/hank/Desktop/123')
#         print(self.data_files)
#         # sort(self.data_files)

#     def __getitem__(self, idx): # All data MUST have the same size before return
#         return st.resize(io.imread('/home/hank/Desktop/123/'+self.data_files[idx]), [20, 20])

#     def __len__(self):
#         return len(self.data_files)






class Resize(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        new_h, new_w = self.output_size
        pad_width = int((new_h - image.shape[1]) / 2)
        resized_image = np.lib.pad(image, ((0,0), (pad_width,pad_width),(pad_width,pad_width)), 'edge')

        return resized_image


class RandomCrop(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_image = image[:, top:top+new_h, left:left+new_w]

        return cropped_image
