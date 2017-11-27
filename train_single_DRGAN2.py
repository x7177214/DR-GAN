#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from util.one_hot import one_hot
from util.Is_D_strong import Is_D_strong
from util.log_learning import log_learning
# from util.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop
from util.DataAugmentation import FaceIdPoseDataset2, Resize, RandomCrop


def train_single_DRGAN(images, id_labels, pose_labels, Nd, Np, Nz, D_model, G_model, args, start_epoch=1):
    '''
    input:
        images: image PATH list
        id_labels / pose_labels: id / pose numpy arr. (Not one hot)
        Nd/Np/Nz: # of id/pose/noise
        D_model: discriminator
        G_model: generator
        args: shell arg.
        start_epoch: starting epoch for training
    '''
    
    if args.cuda:
        D_model.cuda()
        G_model.cuda()

    D_model.train()
    G_model.train()

    lr_Adam    = args.lr
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2
    rndcrop_size = args.rndcrop_train_img_size
    eps = 10**-300 # for safe logarithm
    REAL_LABEL = 0.9 # for LSGAN

    image_size = len(images)
    epoch_time = np.ceil(image_size / args.batch_size).astype(int)

    optimizer_D = optim.Adam(D_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    optimizer_G = optim.Adam(G_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    loss_criterion = nn.CrossEntropyLoss()

    loss_log = []
    steps = 0

    flag_D_strong  = False
    for epoch in range(start_epoch, args.epochs+1):

        # Load augmented data (using img path)
        # transformed_dataset = FaceIdPoseDataset2(images, id_labels, pose_labels,
        #                                 transform = transforms.Compose([RandomCrop((rndcrop_size, rndcrop_size))]), img_size=args.train_img_size)
        transformed_dataset = FaceIdPoseDataset2(images, id_labels, pose_labels,
                                        transform = None, img_size=args.train_img_size)

        dataloader = DataLoader(transformed_dataset, batch_size = args.batch_size, shuffle=True, num_workers=8)

        for i, batch_data in enumerate(dataloader):

            batch_image = torch.FloatTensor(batch_data[0].float())
            batch_id_label = batch_data[1]
            batch_pose_label = batch_data[2]
            minibatch_size = len(batch_image)

            # ノイズと姿勢コードを生成
            if args.noise_type == 0:
                fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))
            elif args.noise_type == 1:
                fixed_noise = torch.FloatTensor(np.random.standard_normal((minibatch_size, Nz)))
            else:
                raise IndexError('current noise_type is not allowed')

            tmp  = torch.LongTensor(np.random.randint(Np, size=minibatch_size))
            pose_code = one_hot(tmp, Np) # Condition 付に使用
            pose_code_label = torch.LongTensor(tmp) # CrossEntropy 誤差に使用

            if args.cuda:
                batch_image, batch_id_label, batch_pose_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_pose_label.cuda()

                fixed_noise, pose_code, pose_code_label = \
                    fixed_noise.cuda(), pose_code.cuda(), pose_code_label.cuda()

            batch_image, batch_id_label, batch_pose_label = \
                Variable(batch_image), Variable(batch_id_label), Variable(batch_pose_label)

            fixed_noise, pose_code, pose_code_label = \
                Variable(fixed_noise), Variable(pose_code), Variable(pose_code_label)

            # Generatorでイメージ生成
            generated = G_model(batch_image, pose_code, fixed_noise)

            steps += 1

            # Dが90%以上の精度の場合は 1:4の比率で学習
            if flag_D_strong:
                num_learning_g = 4
            else:
                num_learning_g = 2

            # Discriminator の学習
            real_output = D_model(batch_image)
            syn_output = D_model(generated.detach()) # .detach() をすることでGeneratorのパラメータを更新しない

            # id,真偽, pose それぞれのロスを計算
            L_id    = loss_criterion(real_output[:, :Nd], batch_id_label)

            if args.use_lsgan:
                L_gan   = Variable.sum((real_output[:, Nd]-REAL_LABEL)**2.0 + (syn_output[:, Nd]-0.0)**2.0) / minibatch_size
            else:
                L_gan   = Variable.sum(real_output[:, Nd].sigmoid().log()*-1 + (1 - syn_output[:, Nd].sigmoid()).log()*-1) / minibatch_size

            L_pose  = loss_criterion(real_output[:, Nd+1:], batch_pose_label)

            d_loss = L_gan + L_id + L_pose

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
                    
            # Generatorの学習
            for k in range(num_learning_g):
                if k > 0: 
                    # Generatorでイメージ生成
                    generated = G_model(batch_image, pose_code, fixed_noise)

                syn_output=D_model(generated)

                # id についての出力と元画像のラベル, 真偽, poseについての出力と生成時に与えたposeコード の ロスを計算
                L_id    = loss_criterion(syn_output[:, :Nd], batch_id_label)

                if args.use_lsgan:
                    L_gan   = Variable.sum((syn_output[:, Nd]-REAL_LABEL)**2.0) / minibatch_size
                else:
                    L_gan   = Variable.sum(syn_output[:, Nd].sigmoid().clamp(min=eps).log()*-1) / minibatch_size
                    
                L_pose  = loss_criterion(syn_output[:, Nd+1:], pose_code_label)

                g_loss = L_gan + L_id + L_pose

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
                
            log_learning(epoch, steps, 'G', g_loss.data[0], args)
            log_learning(epoch, steps, 'D', d_loss.data[0], args) 
            # Discriminator の強さを判別
            flag_D_strong = Is_D_strong(real_output, syn_output, batch_id_label, batch_pose_label, Nd)

        # エポック毎にロスの保存
        loss_log.append([epoch, d_loss.data[0], g_loss.data[0]])

        if epoch%args.save_freq == 0:
            # 各エポックで学習したモデルを保存
            if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            save_path_D = os.path.join(args.save_dir,'epoch{}_D.pt'.format(epoch))
            torch.save(D_model, save_path_D)
            save_path_G = os.path.join(args.save_dir,'epoch{}_G.pt'.format(epoch))
            torch.save(G_model, save_path_G)
            # 最後のエポックの学習前に生成した画像を１枚保存（学習の確認用）
            save_generated_image = generated[0].cpu().data.numpy().transpose(1, 2, 0)
            save_generated_image = np.squeeze(save_generated_image)

            # min~max -> 0~255
            save_generated_image -= save_generated_image.min()
            save_generated_image = save_generated_image/save_generated_image.max()
            save_generated_image = save_generated_image*255.0

            # save_generated_image = (save_generated_image+1)/2.0 * 255.
            save_generated_image = save_generated_image[:,:,[2,1,0]] # convert from BGR to RGB
            save_path_image = os.path.join(args.save_dir, 'epoch{}_generatedimage.png'.format(epoch))
            misc.imsave(save_path_image, save_generated_image.astype(np.uint8))


    # 学習終了後に，全エポックでのロスの変化を画像として保存
    loss_log = np.array(loss_log)
    plt.plot(loss_log[:,1], label="Discriminative Loss")
    plt.plot(loss_log[:,2], label="Generative Loss")
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    filename = os.path.join(args.save_dir, 'Loss_log.png')
    plt.savefig(filename, bbox_inches='tight')
