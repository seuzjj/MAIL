import torch
import torch.nn as nn
from .dnCNN import DnCNN
from .dawn import LevelDAWN
import math


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class unet2d(nn.Module):
    def __init__(self, ch_in, ch_out, ch_num):
        super(unet2d, self).__init__()
        self.numberchannel = ch_num
        self.ch_in = ch_in
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=ch_in, ch_out=self.numberchannel)
        self.Conv2 = conv_block(ch_in=self.numberchannel, ch_out=2 * self.numberchannel)
        self.Conv3 = conv_block(ch_in=2 * self.numberchannel, ch_out=4 * self.numberchannel)
        self.Conv4 = conv_block(ch_in=4 * self.numberchannel, ch_out=8 * self.numberchannel)
        self.Conv5 = conv_block(ch_in=8 * self.numberchannel, ch_out=16 * self.numberchannel)

        self.Up5 = up_conv(ch_in=16 * self.numberchannel, ch_out=8 * self.numberchannel)
        self.Up_conv5 = conv_block(ch_in=16 * self.numberchannel, ch_out=8 * self.numberchannel)

        self.Up4 = up_conv(ch_in=8 * self.numberchannel, ch_out=4 * self.numberchannel)
        self.Up_conv4 = conv_block(ch_in=8 * self.numberchannel, ch_out=4 * self.numberchannel)

        self.Up3 = up_conv(ch_in=4 * self.numberchannel, ch_out=2 * self.numberchannel)
        self.Up_conv3 = conv_block(ch_in=4 * self.numberchannel, ch_out=2 * self.numberchannel)

        self.Up2 = up_conv(ch_in=2 * self.numberchannel, ch_out=self.numberchannel)
        self.Up_conv2 = conv_block(ch_in=2 * self.numberchannel, ch_out=self.numberchannel)

        self.Conv_1x1 = nn.Conv2d(self.numberchannel, ch_in, kernel_size=1, stride=1, padding=0)
        self.fusion_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        x5 = torch.cat((x4, self.Up5(x5)), dim=1)
        x5 = self.Up_conv5(x5)

        # d4 = self.Up4(d5)
        # d4 = self.Up4(x4)
        x4 = torch.cat((x3, self.Up4(x5)), dim=1)
        x4 = self.Up_conv4(x4)

        # d3 = self.Up3(d4)
        x3 = torch.cat((x2, self.Up3(x4)), dim=1)
        x3 = self.Up_conv3(x3)

        # d2 = self.Up2(d3)
        x2 = torch.cat((x1, self.Up2(x3)), dim=1)
        x2 = self.Up_conv2(x2)

        x1 = self.Conv_1x1(x2)
        x = x1 + x
        if self.ch_in > 1:
            x = self.fusion_conv(x)

        return x


class Xnet(nn.Module):
    def __init__(self, channel):
        super(Xnet, self).__init__()
        self.channels = channel
        self.block = DnCNN(in_nc=self.channels, out_nc=self.channels, nc=32, nb=5, act_mode='BR')

    def forward(self, input):
        X = self.block(input)
        # X = F.relu(X+input)
        return X


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class Finalnet(nn.Module):
    def __init__(self, args, ch_in=1):
        super(Finalnet, self).__init__()
        self.num_feat = 64
        self.num_out_ch = 1
        # self.net1 = unet2d(ch_in=ch_in * 4, ch_out=int(ch_in * 4/2), ch_num=4)
        self.prox_u = unet2d(ch_in=1, ch_out=1, ch_num=16)
        self.xfm1 = LevelDAWN(in_planes=1,
                            lifting_size=[2,1], kernel_size=4,  no_bottleneck=True,
                            share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)
        self.xfm2 = LevelDAWN(in_planes=1,
                             lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                             share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)
        self.xfm3 = LevelDAWN(in_planes=1,
                              lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                              share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)
        self.xfm4 = LevelDAWN(in_planes=1,
                              lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                              share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)
        self.prox_A = Xnet(channel=4)
        self.prox_x = Xnet(channel=4)

        self.imf1 = nn.Sequential(nn.Sequential(nn.Conv2d(args.h, self.num_feat, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True)),
                                  Upsample(2, self.num_feat),
                                  nn.Conv2d(self.num_feat, self.num_out_ch, 3, 1, 1))

        self.imf2 = nn.Sequential(nn.Sequential(nn.Conv2d(args.h, self.num_feat, 3, 1, 1),
                                                nn.LeakyReLU(inplace=True)),
                                  Upsample(2, self.num_feat),
                                  nn.Conv2d(self.num_feat, self.num_out_ch, 3, 1, 1))

        self.eta1 = torch.Tensor([args.eta1]).cuda()
        self.eta2 = torch.full([1,4,1,1],args.eta2).cuda()
        self.eta3 = torch.full([1,4,1,1],args.eta3).cuda()
        self.alpha = torch.Tensor([args.alpha]).cuda()
        self.beta = torch.Tensor([args.beta]).cuda()

    def forward(self, Xma, X, U, A):
        xl,r, xh = self.xfm1(X)
        x_w = torch.cat((xl,xh),1)
        Xmal,r, Xmah = self.xfm2(Xma)
        X_w = torch.cat((Xmal, Xmah),1)
        ul, r, uh = self.xfm3(U)
        u_w = torch.cat((ul, uh), 1)
        Al, r, Ah = self.xfm4(A)
        A_w = torch.cat((Al,Ah),1)

        Aw_mid = (1 - 2 * self.eta3) * A_w-2 * 2 * self.eta3 * x_w + 2 * self.eta3 * X_w
        A_w = self.prox_A(Aw_mid)

        xw_mid = (1-2*self.eta2)*x_w + (2*self.eta2/(self.alpha+self.beta))*\
                 (self.alpha * X_w - self.alpha * A_w + self.beta * u_w)
        x_w = self.prox_x(xw_mid)

        X = self.imf1(x_w)
        A = self.imf2(A_w)
        uw_mid = (1-2*self.eta1) * U + 2*self.eta1*X
        U = self.prox_u(uw_mid)
        return X, U, A


class IterWave(nn.Module):
    def __init__(self, args, ch_in=1):
        super(IterWave, self).__init__()
        self.num_layers = args.T
        self.num_feat = 64
        self.num_out_ch = 1
        self.net1 = unet2d(ch_in=8, ch_out=4, ch_num=4)
        self.U = unet2d(ch_in=3, ch_out=1, ch_num=8)
        self.xfm1 = LevelDAWN(in_planes=2,
                              lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                              share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)
        self.prox_x = Xnet(channel=4)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(args.h, self.num_feat, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(2, self.num_feat)
        self.conv_last = nn.Conv2d(self.num_feat, self.num_out_ch, 3, 1, 1)

        self.layer = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Finalnet(args, ch_in=2)
            self.layer.append(layer)

    def forward(self, Xma, XLI):
        # initialization
        x = torch.cat((Xma, XLI), 1)
        xl, r, xh = self.xfm1(x)
        x_w = torch.cat((xl, xh), 1)
        x_w = self.net1(x_w)
        x_w = self.prox_x(x_w)
        X = self.conv_before_upsample(x_w)
        X = self.upsample(X)
        X = self.conv_last(X)
        u0 = torch.cat((X, x), 1)
        U = self.U(u0)
        A = Xma - U

        for layer in self.layer:
            X, U, A = layer(Xma, X, U, A)
        return U