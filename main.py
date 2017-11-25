#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import model.single_DR_GAN_model as single_model
from util.create_randomdata import create_randomdata
from train_single_DRGAN import train_single_DRGAN
from train_multiple_DRGAN import train_multiple_DRGAN
from Generate_Image import Generate_Image2
from data_io import read_path_and_label
import pdb

### controller ###
NUM_TEST_IMG = 10
##################

# NUM_TOTAL_IMG = 18420
NUM_ID = 346
NUM_ILLUMINATION = 20
NUM_SESS = 4

def DataLoader2(data_place):
    """
    ### ouput
    imgs_path_list : N x string; list of image path
    labels_ID : N x 1
    labels_illu : N x 1
    Nd : the nuber of ID in the data
    Ni : the number of discrete pose in the data
    Nz : size of noise vector (Default in the paper is 50)
    """

    Nz = 50
    Ni = NUM_ILLUMINATION
    Nd = NUM_ID
    channel_num = 3

    imgs_path_list, labels_ID, labels_illu = read_path_and_label(data_place)

    return [imgs_path_list, labels_ID, labels_illu, Nd, Ni, Nz, channel_num]

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning & saving parameterss
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=1000, help='number of epochs for train [default: 1000]')
    parser.add_argument('-batch-size', type=int, default=8, help='batch size for training [default: 8]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-save-freq', type=int, default=10, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    # data souce
    parser.add_argument('-random', action='store_true', default=False, help='use randomely created data to run program')
    # parser.add_argument('-data_place', type=str, default='../dataset/cfp-dataset', help='prepared data path to run program')
    # parser.add_argument('-data_place', type=str, default='/Disk2/Multi-Pie/data', help='prepared data path to run program')
    parser.add_argument('-data_place', type=str, default='../../../Multi-Pie/data', help='prepared data path to run program')
    
    # model
    parser.add_argument('-multi-DRGAN', action='store_true', default=False, help='use multi image DR_GAN model')
    parser.add_argument('-images-perID', type=int, default=0, help='number of images per person to input to multi image DR_GAN')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
    parser.add_argument('-g', action='store_true', default=None, help='Generate pose modified image from given image')
    parser.add_argument('-train_img_size', type=int, default=256, help='Image size for training')
    parser.add_argument('-rndcrop_train_img_size', type=int, default=240, help='Random cropped image size for training. Must be 16 * K')

    args = parser.parse_args()

    # create ckpt folder name
    if not(args.g):
        if args.multi_DRGAN:
            args.save_dir = os.path.join(args.save_dir, 'Multi',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            args.save_dir = os.path.join(args.save_dir, 'Single',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(args.save_dir)

    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        text ="\t{}={}\n".format(attr.upper(), value)
        print(text)
        with open('{}/Parameters.txt'.format(args.save_dir),'a') as f:
            f.write(text)

    # input data
    if args.random:
        images, id_labels, pose_labels, Nd, Ni, Nz, channel_num = create_randomdata()
    else:
        print('\n Loading data from [%s]...' % args.data_place)
        try:
            # images, id_labels, pose_labels, Nd, Ni, Nz, channel_num = DataLoader(args.data_place)
            train_img_path_list, id_labels, pose_labels, Nd, Ni, Nz, channel_num = DataLoader2(args.data_place)
        except:
            print("Sorry, failed to load data")

    test_img_path_list = train_img_path_list[-NUM_TEST_IMG:]
    train_img_path_list = train_img_path_list[:-NUM_TEST_IMG]

    # model
    if args.snapshot is None:
        if not(args.multi_DRGAN):
            D = single_model.Discriminator(Nd, Ni, channel_num, args)
            G = single_model.Generator(Ni, Nz, channel_num, args)
            start_epoch = 1
        else:
            if args.images_perID==0:
                print("Please specify -images-perID of your data to input to multi_DRGAN")
                exit()
            # else:
                # D = multi_model.Discriminator(Nd, Ni, channel_num)
                # G = multi_model.Generator(Ni, Nz, channel_num, args.images_perID)
    else:
        print('\n Loading model from [%s]...' % args.snapshot)
        try:
            D = torch.load('{}_D.pt'.format(args.snapshot))
            G = torch.load('{}_G.pt'.format(args.snapshot))
            start_epoch = int(args.snapshot.split('/')[-1][5:])
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()

    if not(args.g):
        if not(args.multi_DRGAN):
            train_single_DRGAN(train_img_path_list, id_labels, pose_labels, Nd, Ni, Nz, D, G, args, start_epoch)
        else:
            if args.batch_size % args.images_perID == 0:
                train_multiple_DRGAN(images, id_labels, pose_labels, Nd, Ni, Nz, D, G, args)
            else:
                print("Please give valid combination of batch_size, images_perID")
                exit()
    else:
        # illu_code = [] # specify arbitrary pose code for every image
        # illu_code = np.random.uniform(-1, 1, (test_img_path_list.shape[0], Ni)) # very noisy code

        # normal illu code
        illu_code = np.zeros((len(test_img_path_list), Ni)) 
        illu_code[:, 7] = 1.0

        # features = Generate_Image(test_img_path_list, illu_code, Nz, G, args)
        features = Generate_Image2(test_img_path_list, illu_code, Nz, G, args)