'''
    This part of code is the pytorch implementation of
    pix2pixGAN, inspired by:
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/generator_model.py
'''
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        # As mentioned in pix2pix paper, each conv block is composed of
        # conv - BN - relu/leakyRelu
        self.conv = nn.Sequential(
            # downsampling conv block.
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down
            # upsampling conv block.
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.fc(x)
        #print("Size of x in FCB: "+str(x.size()))
        # Resize the tensor into shape of (channels , 1, 1)
        x = x.view(x.size(0), -1, 1, 1)
        if x.size(0) == 1:
            layernorm = nn.LayerNorm(x.shape[1:4]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            x = layernorm(x)
        else:
            x = self.bn(x)

        return self.relu(x)

class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=3, features=64, label=None):
        super().__init__()
        # BatchNorm is not applied to the first layer.
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.fc = FCBlock(in_channels=label.size(1), out_channels=features * 8)  # in_channels = label.size(1)
        self.label = label

        # ReLu in decoder are not leaky.
        self.up1 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def assign_label(self, label):
        self.label = label

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)
        label = self.fc(self.label)
        # print(bottleneck.size())
        # print(label.size())

        # Concatenate label and bottleneck output.

        up1 = self.up1(torch.cat([bottleneck, label], 1))
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


# def test():
#     x = torch.randn((1, 3, 256, 256))
#     label = torch.randn((1, 10))
#     model = Pix2PixGenerator(in_channels=3, features=64, label=label)
#     preds = model(x)
#     print(preds.shape)
#
#
# test()