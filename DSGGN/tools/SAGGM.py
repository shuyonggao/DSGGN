
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch)  # , eps=0.001

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

    def initialize(self):
        for n, m in self.named_children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                print('部分元素没有初始化')

class SAGGM(nn.Module):
    def __init__(self, guide_low_ch):
        super(SAGGM, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(guide_low_ch, guide_low_ch, kernel_size=3, padding=1),
                             nn.BatchNorm2d(guide_low_ch),
                             nn.ReLU(inplace=True))

    def forward(self, guide_x, low_x1, low_x2=None):
        guide_shape = guide_x.size()[2:]
        low_shape = low_x1.size()[2:]
        low_pool1 = nn.AdaptiveMaxPool2d(guide_shape)(low_x1)
        if low_x2 is not None:
            low_pool2 = nn.AdaptiveMaxPool2d(guide_shape)(low_x2)
            guide = guide_x + low_pool1 + low_pool2
        else:
            guide = guide_x + low_pool1
        guide = self.conv(guide)
        guide = F.interpolate(guide, size=low_shape, mode='bilinear', align_corners=True)
        return guide

class SAGGM_v2(nn.Module):
    def __init__(self, guide_low_ch, low_x2=False):
        super(SAGGM_v2, self).__init__()

        self.down_conv_x1 = BasicConv2d(guide_low_ch, guide_low_ch, kernel_size=3, padding=1) # 下采样之后的特征进行一次卷积
        if low_x2 == True:
            self.down_conv_x2 = BasicConv2d(guide_low_ch, guide_low_ch, kernel_size=3, padding=1)

        self.down_add_up = BasicConv2d(guide_low_ch, guide_low_ch, kernel_size=3, padding=1) #下采样相加之后把自己上采样，再进行一次卷积，然后再下采样回来

        self.add_bn = nn.BatchNorm2d(guide_low_ch)

        self.conv = nn.Sequential(nn.Conv2d(guide_low_ch, guide_low_ch, kernel_size=3, padding=1),
                             nn.BatchNorm2d(guide_low_ch),
                             nn.ReLU(inplace=True))
        # 最后

    def forward(self, guide_x, low_x1, low_x2=None):
        guide_shape = guide_x.size()[2:]
        low_shape = low_x1.size()[2:]
        low_pool1 = nn.AdaptiveMaxPool2d(guide_shape)(low_x1)
        if low_x2 is not None:
            low_pool2 = nn.AdaptiveMaxPool2d(guide_shape)(low_x2)
            low_pool1 = self.down_conv_x1(low_pool1)
            low_pool2 = self.down_conv_x2(low_pool2)
            guide = guide_x + low_pool1 + low_pool2
        else:
            low_pool1 = self.down_conv_x1(low_pool1)
            guide = guide_x + low_pool1

        guide_up = F.interpolate(guide, size=low_shape, mode='bilinear', align_corners=True)
        guide_up = self.down_add_up(guide_up)
        guide_up = nn.AdaptiveMaxPool2d(guide_shape)(guide_up)

        guide = guide + guide_up

        guide = F.relu(self.add_bn(guide), inplace=True)
        guide = F.interpolate(guide, size=low_shape, mode='bilinear', align_corners=True)
        guide = self.conv(guide)
        return guide


if __name__== '__main__':
    input_g = torch.randn(3,64,9,9)
    input_l = torch.randn(3,64,256,256)
    net = SAGGM_v2(64)
    out = net(input_g, input_l)
    print(out.size())
