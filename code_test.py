from torch import nn, optim
import torch
from torch.autograd import Variable
import numpy as np

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

        num_conv = 8
        num_basis_channel = 64
        in_channel = num_basis_channel

        # create the encoder pathway and add to a list
        for i in range(num_conv):
            if i == 0:
                down_conv=nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)),
                    nn.Conv2d(channel_num, 64, 3, 2, 0, bias=False),
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
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout(0.5),
                )
            else:
                up_conv=nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channel*2, out_channel, 3, 2, 0, bias=False), 
                    Crop([0, 1, 0, 1]),                                
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout(0.5),
                )
            in_channel = out_channel

            self.up_convs.append(up_conv)

        # final layer
        up_conv=nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 3, 2, 0, bias=False),   
            Crop([0, 1, 0, 1]),                                
            nn.Tanh(),
        )
        self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

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
        for i, (module, module2) in enumerate(zip(self.up_convs, self.down_convs)):
            # print(xx[0])
            # print(xx[1])
            # print(xx[2])
            if i == 0: 
                print(module, module2)
            x = module(x)
            if i < 7:
                enc_out = encoder_outs[-(i+2)]
                x = torch.cat([x, enc_out], 1)

        return x


Np = 20
Nz = 15
channel_num = 3
G = Generator(Np, Nz, channel_num)

import skimage.io as io
import skimage.transform as st

pose = torch.FloatTensor(np.random.uniform(-1, 1, (1, Np)))
noise = torch.FloatTensor(np.random.uniform(-1, 1, (1, Nz)))

img = io.imread('/home/hank/Desktop/2.jpg')
img = st.resize(img, [256, 256])
img = img.transpose(2, 0, 1)

img = Variable(torch.FloatTensor(img))
img = img.view(-1, 3, 256, 256)

print(G(img, pose, noise))