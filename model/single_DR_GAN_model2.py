#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn, optim
from torch.autograd import Variable
import pdb

'''
pix2pix version arch.
'''

class Discriminator(nn.Module):
    """
    multi-task CNN for identity and pose classification

    ### init
    Nd : Number of identitiy to classify
    Np : Number of pose to classify

    """

    def __init__(self, Nd, Np, channel_num, args=None):
        super(Discriminator, self).__init__()

        convLayers = [
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bxchx256x256 -> Bxchx257x257
            nn.Conv2d(channel_num, 64, 3, 2, 0, bias=False), # Bxchx257x257 -> Bx64x128x128
            nn.LeakyReLU(0.2, inplace=True),

            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x128x128 -> Bx64x129x129
            nn.Conv2d(64, 128, 3, 2, 0, bias=False),         # Bx64x129x129 -> Bx128x64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x64x64 -> Bx128x65x65
            nn.Conv2d(128, 256, 3, 2, 0, bias=False),        # Bx128x65x65 -> Bx256x32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x32x32 -> Bx256x33x33
            nn.Conv2d(256, 512, 3, 2, 0, bias=False),        # Bx256x33x33 -> Bx512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fc = nn.Linear(512*16*16, Nd+1+Np)

        # 重みは全て N(0, 0.02) で初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        # 畳み込み -> 平均プーリングの結果 B x 320 x 1 x 1の出力を得る
        x = self.convLayers(input)

        x = x.view(-1, 512*16*16)

        # 全結合
        x = self.fc(x) # Bx(512x16x16) -> B x (Nd+1+Np)

        return x

## nn.Module を継承しても， super でコンストラクタを呼び出さないと メンバ変数 self._modues が
## 定義されずに後の重み初期化の際にエラーを出す
## sef._modules はモジュールが格納するモジュール名を格納しておくリスト

class Crop(nn.Module):
    """
    Generator でのアップサンプリング時に， ダウンサンプル時のZeroPad2d と逆の事をするための関数
    論文著者が Tensorflow で padding='SAME' オプションで自動的にパディングしているのを
    ダウンサンプル時にはZeroPad2dで，アップサンプリング時には Crop で実現

    ### init
    crop_list : データの上下左右をそれぞれどれくらい削るか指定
    """

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        # crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
        self.crop_list = crop_list

    def forward(self, x):
        B,C,H,W = x.size()
        x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

        return x

class Generator(nn.Module):
    """
    Encoder/Decoder conditional GAN conditioned with pose vector and noise vector

    ### init
    Np : Dimension of pose vector (Corresponds to number of dicrete pose classes of the data)
    Nz : Dimension of noise vector

    """

    def __init__(self, Np, Nz, channel_num, args=None):
        super(Generator, self).__init__()
        self.features = []

        self.down_convs = []
        self.up_convs = []
        self.bn_dropouts = []

        num_conv = 8
        num_basis_channel = 64
        in_channel = num_basis_channel

        # create the encoder pathway and add to a list
        for i in range(num_conv):
            if i == 0:
                down_conv=nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)),
                    nn.Conv2d(channel_num, num_basis_channel, 3, 2, 0, bias=False),
                )
            else:
                out_channel = in_channel * 2
                if out_channel >= 512: out_channel = 512
                down_conv=nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ZeroPad2d((0, 1, 0, 1)),                      
                    nn.Conv2d(in_channel, out_channel, 3, 2, 0, bias=False),     
                    nn.BatchNorm2d(out_channel),
                )
                in_channel = out_channel
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        in_channel = out_channel
        for i in range(num_conv-1):
            if i > 3: out_channel = in_channel // 2

            if i == 0:
                up_conv=nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 0, bias=False), 
                    Crop([0, 1, 0, 1]),                                
                )
            else: # start to concat the down_conv
                up_conv=nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channel*2, out_channel, 3, 2, 0, bias=False), 
                    Crop([0, 1, 0, 1]),                                
                )
            bn_dropout=nn.Sequential(
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout(0.5),
            )      
            
            in_channel = out_channel

            self.up_convs.append(up_conv)
            self.bn_dropouts.append(bn_dropout)

        # final layer
        self.final_up_conv=nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_basis_channel*2, channel_num, 3, 2, 0, bias=False),   
            Crop([0, 1, 0, 1]),                                
            nn.Tanh(),
        )

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.bn_dropouts = nn.ModuleList(self.bn_dropouts)

        self.G_dec_fc = nn.Linear(512+Np+Nz, 512*1*1)

        # 重みは全て N(0, 0.02) で初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input, pose, noise):

        encoder_outs = []

        # encoder pathway, save outputs for merging
        # Bxchx256x256 -> Bx512x1x1
        for i, module in enumerate(self.down_convs):
            if i == 0:
                x = module(input)
            else:
                x = module(x)
            encoder_outs.append(x)
        
        x = x.view(-1, 512)                 # Bx512x1x1 -> Bx512
        self.features = x
        x = torch.cat([x, pose, noise], 1)  # Bx512 -> B x (512+Np+Nz)
        x = self.G_dec_fc(x)                # B x (512+Np+Nz) -> B x (512)
        x = x.view(-1, 512, 1, 1)           # B x (512) -> Bx512x1x1

        # Bx512x1x1 -> Bxchx256x256
        for i, (module, bn_dropout) in enumerate(zip(self.up_convs, self.bn_dropouts)):

            x = module(x)
            x = x.contiguous() # enable bn in decoder path
            x = bn_dropout(x)

            enc_out = encoder_outs[-(i+2)]
            x = torch.cat([x, enc_out], 1)

        x = self.final_up_conv(x)

        return x
