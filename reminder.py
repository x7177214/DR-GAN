G_enc_convLayers = [
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx3x256x256 -> Bx3x257x257
            nn.Conv2d(channel_num, 64, 3, 2, 0, bias=False), # Bx3x257x257 -> Bx64x128x128

            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x128x128-> Bx64x129x129
            nn.Conv2d(64, 128, 3, 2, 0, bias=False),         # Bx64x129x129-> Bx128x64x64
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
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx512x16x16 -> Bx512x17x17
            nn.Conv2d(512, 512, 3, 2, 0, bias=False),        # Bx512x17x17 -> Bx512x8x8
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx512x8x8 -> Bx512x9x9
            nn.Conv2d(512, 512, 3, 2, 0, bias=False),        # Bx512x9x9 -> Bx512x4x4
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx512x4x4 -> Bx512x5x5
            nn.Conv2d(512, 512, 3, 2, 0, bias=False),        # Bx512x5x5 -> Bx512x2x2
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx512x2x2 -> Bx512x3x3
            nn.Conv2d(512, 512, 3, 2, 0, bias=False),        # Bx512x3x3 -> Bx512x1x1
            nn.BatchNorm2d(512),
        ]


        G_dec_convLayers = [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 3, 2, 0, bias=False), # Bx512x1x1 -> Bx512x3x3
            Crop([0, 1, 0, 1]),                                # Bx512x3x3 -> Bx512x2x2
            nn.BatchNorm2d(512),
            nn.dropout(0.5),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 3, 2, 0, bias=False), # Bx512x2x2 -> Bx512x5x5
            Crop([0, 1, 0, 1]),                                # Bx512x5x5 -> Bx512x4x4
            nn.BatchNorm2d(512),
            nn.dropout(0.5),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 3, 2, 0, bias=False), # Bx512x4x4 -> Bx512x9x9
            Crop([0, 1, 0, 1]),                                # Bx512x9x9 -> Bx512x8x8
            nn.BatchNorm2d(512),
            nn.dropout(0.5),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 3, 2, 0, bias=False), # Bx512x8x8 -> Bx512x17x17
            Crop([0, 1, 0, 1]),                                # Bx512x17x17 -> Bx512x16x16
            nn.BatchNorm2d(512),
            nn.dropout(0.5),      

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 3, 2, 0, bias=False), # Bx512x16x16 -> Bx256x33x33
            Crop([0, 1, 0, 1]),                                # Bx256x33x33 -> Bx256x32x32
            nn.BatchNorm2d(256),
            nn.dropout(0.5),     

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2, 0, bias=False), # 128x65
            Crop([0, 1, 0, 1]),                                # 128x64
            nn.BatchNorm2d(128),
            nn.dropout(0.5), 

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 0, bias=False),  # 64x129
            Crop([0, 1, 0, 1]),                                # 64x128
            nn.BatchNorm2d(64),
            nn.dropout(0.5), 

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 3, 2, 0, bias=False),    # 3x257
            Crop([0, 1, 0, 1]),                                # 3x256
   
            nn.Tanh(),
        ]