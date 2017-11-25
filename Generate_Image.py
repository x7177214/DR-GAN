#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import skimage
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable

def Generate_Image(images, pose_code, Nz, G_model, args):
    """
    Generate_Image with learned Generator

    ### input
    images      : source images
    pose_code   : vector which specify pose to generate image from source image
    Nz          : size of noise vecotr
    G_model     : learned Generator
    args        : options

    ### output
    features    : extracted disentangled features of each image

    """
    if args.cuda:
        # D_model.cuda()
        G_model.cuda()

    G_model.eval()

    image_size = images.shape[0]
    epoch_time = np.ceil(image_size / args.batch_size).astype(int)
    features = []
    image_number = 1

    if not(args.multi_DRGAN):

        for i in range(epoch_time):
            start = i*args.batch_size
            end = start + args.batch_size
            batch_image = torch.FloatTensor(images[start:end])
            batch_pose_code = torch.FloatTensor(pose_code[start:end]) # Condition 付に使用
            minibatch_size = len(batch_image)

            fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (minibatch_size, Nz)))

            if args.cuda:
                batch_image, fixed_noise, batch_pose_code = \
                    batch_image.cuda(), fixed_noise.cuda(), batch_pose_code.cuda()

            batch_image, fixed_noise, batch_pose_code = \
                Variable(batch_image), Variable(fixed_noise), Variable(batch_pose_code)

            # Generatorでイメージ生成
            generated = G_model(batch_image, batch_pose_code, fixed_noise)
            features.append(G_model.features)

            # バッチ毎に生成したイメージを
            for j in range(minibatch_size):
                save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                save_generated_image = np.squeeze(save_generated_image)
                save_generated_image = (save_generated_image+1)/2.0 * 255.
                save_generated_image = save_generated_image[:,:,[2,1,0]] # convert from BGR to RGB
                save_dir = '{}_generated'.format(args.snapshot)
                filename = os.path.join(save_dir, '{}.jpg'.format(str(image_number)))
                if not os.path.isdir(save_dir): os.makedirs(save_dir)
                print('saving {}'.format(filename))
                misc.imsave(filename, save_generated_image.astype(np.uint8))

                image_number += 1

        features = torch.cat(features)

    else:

        for i in range(epoch_time):
            start = i*args.batch_size
            end = start + args.batch_size
            batch_image = torch.FloatTensor(images[start:end])
            batch_pose_code = torch.FloatTensor(pose_code[start:end]) # Condition 付に使用
            batch_pose_code_unique = torch.FloatTensor(batch_pose_code[::args.images_perID])
            minibatch_size_unique = len(batch_image) // args.images_perID

            fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (minibatch_size_unique, Nz)))

            if args.cuda:
                batch_image, fixed_noise, batch_pose_code_unique = \
                    batch_image.cuda(), fixed_noise.cuda(), batch_pose_code_unique.cuda()

            batch_image, fixed_noise, batch_pose_code_unique = \
                Variable(batch_image), Variable(fixed_noise), Variable(batch_pose_code_unique)

            # Generatorでイメージ生成
            generated = G_model(batch_image, batch_pose_code_unique, fixed_noise)
            features.append(G_model.features)

            # バッチ毎に生成したイメージを
            for j in range(minibatch_size_unique):
                save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                save_generated_image = np.squeeze(save_generated_image)
                save_generated_image = (save_generated_image+1)/2.0 * 255.
                save_generated_image = save_generated_image[:,:,[2,1,0]] # convert from BGR to RGB
                save_dir = '{}_generated'.format(args.snapshot)
                filename = os.path.join(save_dir, '{}.jpg'.format(str(image_number)))
                if not os.path.isdir(save_dir): os.makedirs(save_dir)
                print('saving {}'.format(filename))
                misc.imsave(filename, save_generated_image.astype(np.uint8))

                image_number += 1

        features = torch.cat(features)
    return features

def Generate_Image2(images_path, pose_code, Nz, G_model, args):
    """
    Generate_Image with learned Generator

    ### input
    images_path : source images path, B x str
    pose_code   : vector which specify pose to generate image from source image
    Nz          : size of noise vecotr
    G_model     : learned Generator
    args        : options

    ### output
    features    : extracted disentangled features of each image

    """
    def read_img_and_preprocessing(images_path):

        # For FaceCrop
        h_start = 50
        w_start = 198
        H = 300
        W = 300

        final_size = args.rndcrop_train_img_size
        num_img = len(images_path)
        num_channel = 3
        images = np.empty((num_img, num_channel, final_size, final_size))

        for i in range(num_img):
            path = images_path[i]

            # read image
            image = skimage.io.imread(path)
            
            # FaceCrop
            image = image[h_start:h_start+H, w_start:w_start+W, :]
            # img = img[50:50+300, 198:198+300, :]

            # Resize
            image = skimage.transform.resize(image, [args.train_img_size, args.train_img_size], mode='reflect')

            #[0,255] -> [-1,1]
            image = image / 255.0
            image = 2.0 * image - 1.0

            # RGB -> BGR
            image = image[:, :, [2, 1, 0]]

            # H x W x C -> C x H x W
            image = image.transpose(2, 0, 1)

            # RandomCrop
            h, w = args.train_img_size
            new_h, new_w = args.rndcrop_train_img_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[:, top:top+new_h, left:left+new_w]

            # collect
            images[i] = image

        return images

    if args.cuda:
        # D_model.cuda()
        G_model.cuda()

    G_model.eval()

    image_size = len(images_path)
    epoch_time = np.ceil(image_size / args.batch_size).astype(int)
    features = []
    image_number = 1

    images = read_img_and_preprocessing(images_path)

    if not(args.multi_DRGAN):

        for i in range(epoch_time):
            start = i*args.batch_size
            end = start + args.batch_size
            batch_image = torch.FloatTensor(images[start:end])
            batch_pose_code = torch.FloatTensor(pose_code[start:end]) # Condition 付に使用
            minibatch_size = len(batch_image)

            fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))

            if args.cuda:
                batch_image, fixed_noise, batch_pose_code = \
                    batch_image.cuda(), fixed_noise.cuda(), batch_pose_code.cuda()

            batch_image, fixed_noise, batch_pose_code = \
                Variable(batch_image), Variable(fixed_noise), Variable(batch_pose_code)

            # Generatorでイメージ生成
            generated = G_model(batch_image, batch_pose_code, fixed_noise)
            features.append(G_model.features)

            # バッチ毎に生成したイメージを
            for j in range(minibatch_size):
                save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                save_generated_image = np.squeeze(save_generated_image)
                save_generated_image = (save_generated_image+1)/2.0 * 255.
                save_generated_image = save_generated_image[:,:,[2,1,0]] # convert from BGR to RGB
                save_dir = '{}_generated'.format(args.snapshot)
                filename = os.path.join(save_dir, '{}.jpg'.format(str(image_number)))
                if not os.path.isdir(save_dir): os.makedirs(save_dir)
                print('saving {}'.format(filename))
                misc.imsave(filename, save_generated_image.astype(np.uint8))

                image_number += 1

        features = torch.cat(features)

    else:

        for i in range(epoch_time):
            start = i*args.batch_size
            end = start + args.batch_size
            batch_image = torch.FloatTensor(images[start:end])
            batch_pose_code = torch.FloatTensor(pose_code[start:end]) # Condition 付に使用
            batch_pose_code_unique = torch.FloatTensor(batch_pose_code[::args.images_perID])
            minibatch_size_unique = len(batch_image) // args.images_perID

            fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (minibatch_size_unique, Nz)))

            if args.cuda:
                batch_image, fixed_noise, batch_pose_code_unique = \
                    batch_image.cuda(), fixed_noise.cuda(), batch_pose_code_unique.cuda()

            batch_image, fixed_noise, batch_pose_code_unique = \
                Variable(batch_image), Variable(fixed_noise), Variable(batch_pose_code_unique)

            # Generatorでイメージ生成
            generated = G_model(batch_image, batch_pose_code_unique, fixed_noise)
            features.append(G_model.features)

            # バッチ毎に生成したイメージを
            for j in range(minibatch_size_unique):
                save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                save_generated_image = np.squeeze(save_generated_image)
                save_generated_image = (save_generated_image+1)/2.0 * 255.
                save_generated_image = save_generated_image[:,:,[2,1,0]] # convert from BGR to RGB
                save_dir = '{}_generated'.format(args.snapshot)
                filename = os.path.join(save_dir, '{}.jpg'.format(str(image_number)))
                if not os.path.isdir(save_dir): os.makedirs(save_dir)
                print('saving {}'.format(filename))
                misc.imsave(filename, save_generated_image.astype(np.uint8))

                image_number += 1
        features = torch.cat(features)
        
    return features