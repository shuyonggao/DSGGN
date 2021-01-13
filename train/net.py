#!/usr/bin/python3
# coding=utf-8


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.non_local_aspp import ASPP, NonLocalASPP
from tools.SAGGM import SAGGM, SAGGM_v2

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, BasicConv2d):
            weight_init(m)

        elif isinstance(m, SAGGM_v2):
            weight_init(m)
        elif isinstance(m, NonLocalASPP):
            weight_init(m)
        elif isinstance(m, ASPP):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        shape = x.size()[2:]
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return [out1, out2, out3, out4, out5], shape

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder5 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.decoder4 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.decoder3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.decoder2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.decoder1 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.saggm5_4 = SAGGM_v2(64)
        self.saggm5_3 = SAGGM_v2(64)
        self.saggm5_2 = SAGGM_v2(64)
        self.saggm5_1 = SAGGM_v2(64)

        self.non_local = NonLocalASPP(64)

    def forward(self, list):

        out5 = self.decoder5(list[0])
        out5 = self.non_local(out5)

        out5_4 = self.saggm5_4(out5, list[1])
        out5_3 = self.saggm5_3(out5, list[2])
        out5_2 = self.saggm5_2(out5, list[3])
        out5_1 = self.saggm5_1(out5, list[0])

        out4 = self.decoder4(list[1] + out5_4)
        out3 = self.decoder3(F.interpolate(out4, size=list[2].size()[2:], mode='bilinear') + list[2] + out5_3)
        out2 = self.decoder2(F.interpolate(out3, size=list[3].size()[2:], mode='bilinear') + list[3] + out5_2)
        #out1 = self.decoder1(F.interpolate(out2, size=list[0].size()[2:], mode='bilinear') + list[0] + out5_1)

        return   out2, out5   # out1,

    def initialize(self):
        weight_init(self)


class Decoder_edge(nn.Module):
    def __init__(self):
        super(Decoder_edge, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(64)

        #self.non_local = NonlocalBlock(64)
        #self.non_local = ChannelGCN(64)
        self.non_local = NonLocalASPP(64)
        self.saggm = SAGGM_v2(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, input):

        out0 = F.relu(self.bn0(self.conv0(input[0])), inplace=True)
        out0 =  self.non_local(out0)
        out0_3 = self.saggm(out0, input[1])

        out0_3 = F.interpolate(out0_3, size=input[1].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input[1] + out0_3)), inplace=True)
        return (out3,  out0)

    def initialize(self):
        weight_init(self)


class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Fuse(nn.Module):
    def __init__(self):
        super(Fuse, self).__init__()
        #self.non_local = NonlocalBlock(64)
        # self.non_local = ChannelGCN(64)
        self.non_local = NonLocalASPP(64)

        self.merge5 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.fuse5 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.merge2 = nn.Conv2d(192, 64, kernel_size=1, bias=False)
        self.fuse2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.saggm5_2 = SAGGM_v2(64, low_x2=True)

    def forward(self, input1, input2):
        fuse5 = self.fuse5(self.merge5(torch.cat((input1[1], input2[1]), dim=1)))  # +decoder_feat5
        fuse5 = self.non_local(fuse5)

        fuse5_2 = self.saggm5_2(fuse5, input1[0], input2[0])
        fuse2 = self.fuse2(self.merge2(torch.cat((input1[0], input2[0], fuse5_2), dim=1)))
        return fuse2

    def initialize(self):
        weight_init(self)


class DSGGN(nn.Module):
    def __init__(self, cfg):
        super(DSGGN, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet(cfg)

        self.conv5b = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decodersal = Decoder()
        self.decoder_edge = Decoder_edge()
        self.fuse = Fuse()

        self.linearb = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.initialize()

    def forward(self, x, shape=None):

        ####-----------fuse---------------------####
        (out1, out2, out3, out4, out5), shape = self.bkbone(x)
        out2b, out3b, out4b, out5b = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)

        outsal  = self.decodersal([out5b, out4b, out3b, out2b])
        outedge = self.decoder_edge([out5b,out2b])

        out1 = self.fuse(outsal, outedge)
        ################################################

        if shape is None:
            shape = x.size()[2:]
        out1 = F.interpolate(self.linear(out1), size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb(outsal[0]), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard(outedge[0]), size=shape, mode='bilinear')

        return outb1, outd1, out1

    def initialize(self):
        if self.cfg.snapshot:
            pass
            #self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time
    import dataset

    cfg = dataset.Config(datapath='/home/gaosy/DATA/GT/ECSSD', snapshot='./out/model-66', mode='test')
    data = dataset.Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)

    res = []
    net = DSGGN(cfg).cuda()
    for image, (H, W), name in loader:
        image = image.cuda().float()
        torch.cuda.synchronize()
        start = time.time()
        predict = net(image)
        torch.cuda.synchronize()
        end = time.time()
        res.append(end - start)
    time_sum = 0
    for i in res:
        time_sum += i
    print("FPS: %f" % (1.0 / (time_sum / len(res))))
