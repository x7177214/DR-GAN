import numpy as np
import os
import random

### controller ###
ROOT_PATH = '/media/hank/ADATA HD710/multi-PIE/Multi-Pie/data'
################## 

NUM_ID = 346
NUM_ILLUMINATION = 20
NUM_SESS = 4

# def read_path_and_label(root_path):
#     '''
#     input: root_path
#     output: 
#         imgs_path_list: 
#             N x string; list of image path
#         labels_ID: 
#             N x NUM_ID; numpy arr of ID one hot vector
#         labels_illu: 
#             N x NUM_ILLUMINATION; numpy arr of illumination one hot vector

#     N: total num. images
#     '01' expression and '05_1' viewpoint is uesd
#     '''
 
#     imgs_path_list = []
#     labels_ID = []
#     labels_illu = []

#     for sess in range(NUM_SESS):
#         sess += 1
#         current_path = root_path + '/session%.2d' % sess + '/multiview/'
#         list_ID = [name for name in os.listdir(current_path) if os.path.isdir(current_path+name)]

#         for person in list_ID: # start from 001
#             # ID one hot vector 
#             one_hot_ID = [0] * NUM_ID
#             one_hot_ID[int(person)-1] = 1

#             for illu in range(NUM_ILLUMINATION):
#                 current_path = root_path + '/session%.2d' % sess + '/multiview/' + person + '/01/05_1/'
#                 file_name = person + '_%.2d' % sess + '_01_051_' + '%.2d' % illu
#                 imgs_path_list.extend([current_path + file_name + '.png'])

#                 # save illu one hot vector 
#                 one_hot_illu = [0] * NUM_ILLUMINATION
#                 one_hot_illu[int(illu)] = 1
#                 labels_illu.append(one_hot_illu)

#                 # save ID one hot vector 
#                 labels_ID.append(one_hot_ID)

#     # shuffle the data
#     num_data = len(imgs_path_list)
#     rnd_s = random.sample(range(num_data), num_data)
#     imgs_path_list = [imgs_path_list[i] for i in rnd_s]
#     labels_ID = [labels_ID[i] for i in rnd_s]
#     labels_illu = [labels_illu[i] for i in rnd_s]

#     # to numpy
#     labels_ID = np.asarray(labels_ID)
#     labels_illu = np.asarray(labels_illu)

#     return imgs_path_list, labels_ID, labels_illu

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

    for sess in range(NUM_SESS):
        sess += 1
        current_path = root_path + '/session%.2d' % sess + '/multiview/'
        list_ID = [name for name in os.listdir(current_path) if os.path.isdir(current_path+name)]

        for person in list_ID: # start from 001
            # ID one hot vector 
            label_ID = int(person)-1

            for illu in range(NUM_ILLUMINATION):
                current_path = root_path + '/session%.2d' % sess + '/multiview/' + person + '/01/05_1/'
                file_name = person + '_%.2d' % sess + '_01_051_' + '%.2d' % illu
                imgs_path_list.extend([current_path + file_name + '.png'])

                # save illu one hot vector 
                label_illu = int(illu)
                labels_illu.append(label_illu)

                # save ID one hot vector 
                labels_ID.append(label_ID)

    # shuffle the data
    num_data = len(imgs_path_list)
    rnd_s = random.sample(range(num_data), num_data)
    imgs_path_list = [imgs_path_list[i] for i in rnd_s]
    labels_ID = [labels_ID[i] for i in rnd_s]
    labels_illu = [labels_illu[i] for i in rnd_s]

    # to numpy
    labels_ID = np.asarray(labels_ID)
    labels_illu = np.asarray(labels_illu)

    return imgs_path_list, labels_ID, labels_illu

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