import sys
sys.path.append('..')
import os
import glob
import numpy as np
from skimage import io, transform
from tqdm import tqdm

'''
input: 
    folder containing image, identity and viewpoint information
output:
    images.npy
    ids.npy
    yaws.npy
'''

### controller ###
image_dir = "../dataset/cfp-dataset/Data/Images/"
LONGSIDE_SIZE = 110
################## 

# CFP の画像を 長辺を指定した長さに， 短辺は 変換後に リサイズするクラス

class Resize(object):
    #  assume image  as H x W x C numpy array
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
        
    def __call__(self, image):
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self.output_size, int(self.output_size * w / h)
        else:
            new_h, new_w = int(self.output_size * h / w), self.output_size

        resized_image = transform.resize(image, (new_h, new_w))
        
        if h>w:
            diff = self.output_size - new_w
            if diff%2 == 0:
                pad_l = int(diff/2)
                pad_s = int(diff/2)
            else:
                pad_l = int(diff/2)+1
                pad_s = int(diff/2)

            padded_image = np.lib.pad(resized_image, ((0,0), (pad_l,pad_s), (0,0)), 'edge')

        else:
            diff = self.output_size - new_h
            if diff%2==0:
                pad_l = int(diff/2)
                pad_s = int(diff/2)
            else:
                pad_l = int(diff/2)+1
                pad_s = int(diff/2)

            padded_image = np.lib.pad(resized_image, ((pad_l,pad_s), (0,0),  (0,0)), 'edge')

        return padded_image

# 画像をロードし，長辺 110pix 短編 110pix になるようにエッジの画素値で padding
rsz = Resize(LONGSIDE_SIZE)

Indv_dir = []
for x in os.listdir(image_dir):
    if os.path.isdir(os.path.join(image_dir, x)):
        Indv_dir.append(x)
        
Indv_dir=np.sort(Indv_dir)

images = np.zeros((7000, LONGSIDE_SIZE, LONGSIDE_SIZE, 3))
id_labels = np.zeros(7000)
pose_labels = np.zeros(7000)
count = 0
gray_count = 0

for i in tqdm(range(len(Indv_dir))):
    Frontal_dir = os.path.join(image_dir, Indv_dir[i], 'frontal')
    Profile_dir = os.path.join(image_dir, Indv_dir[i], 'profile')
    
    front_img_files = os.listdir(Frontal_dir)
    prof_img_files = os.listdir(Profile_dir)
    
    for img_file in front_img_files:
        img = io.imread(os.path.join(Frontal_dir, img_file))
        if len(img.shape)==2:
            gray_count = gray_count+1
            continue
        img_rsz = rsz(img)
        images[count] = img_rsz
        id_labels[count] = i
        pose_labels[count] = 0
        count = count + 1
    
    for img_file in prof_img_files:
        img = io.imread(os.path.join(Profile_dir, img_file))
        if len(img.shape)==2:
            gray_count = gray_count+1
            continue
        img_rsz = rsz(img)
        images[count] = img_rsz
        id_labels[count] = i
        pose_labels[count] = 1
        count = count + 1
    
id_labels = id_labels.astype('int64')
pose_labels = pose_labels.astype('int64')

#[0,255] -> [-1,1]
images = images * 2.0 - 1.0
# RGB -> BGR
images = images[:,:,:,[2,1,0]]
# B x H x W x C-> B x C x H x W
images = images.transpose(0, 3, 1, 2)

# 白黒画像データを取り除く
images = images[:gray_count*-1]
id_labels = id_labels[:gray_count*-1]
pose_labels = pose_labels[:gray_count*-1]

np.save('images', images)
np.save('ids', id_labels)
np.save('yaws', pose_labels)
