import numpy as np
import os
import random
import skimage.io 
import skimage.transform

### controller ###
ROOT_PATH = '/home/hank/Desktop/test'
################## 

NUM_ID = 346
NUM_ILLUMINATION = 20
NUM_CASE = 20

def read_path_and_label(root_path):
    '''
    input: root_path
    output: 
        imgs_path_list: 
            N x string; list of image path
        labels_ID: 
            N x NUM_ID; numpy arr of ID one hot vector
        labels_illu: 
            N x NUM_ILLUMINATION; numpy arr of illumination one hot vector

    N: total num. images
    '01' expression and '05_1' viewpoint is uesd
    '''
 
    imgs_path_list = []
    labels_ID = []
    labels_illu = []


    for case_id in range(3):

        current_path = root_path + '/case%d/' % case_id
        source_list_ID = [name for name in os.listdir(current_path) if os.path.isfile(current_path+name)]

        for i, sample in enumerate(source_list_ID): # start from 001
            imgs_path_list.append(current_path+sample)

    print(imgs_path_list)

    return imgs_path_list

if __name__ == '__main__':
    imgs_path_list, labels_ID, labels_illu= read_path_and_label(ROOT_PATH)
    # a = np.random.randint(len(imgs_path_list))
    # print(len(imgs_path_list))
    # print(imgs_path_list[5])
    # print(np.where(labels_ID[5]==1))
    # print(np.where(labels_illu[5]==1))

    import skimage.io as io
    import matplotlib.pyplot as plt

    # find mean face
    # imgs = 0
    # for i in range(len(imgs_path_list)):
    #     img = io.imread(imgs_path_list[i])/255.0
    #     imgs += img
    for i in range(10):

        img = io.imread(imgs_path_list[i])/255.0
        print(img.shape)
        img = img[50:50+300, 198:198+300, :]
        
        plt.imshow(img)
        plt.show()