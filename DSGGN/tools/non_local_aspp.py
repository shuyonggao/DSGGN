
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
使用多个空洞卷积，不同尺度的高层语义特征识别能力
'''

class NonlocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels = None, bn_layer=True):
        super(NonlocalBlock, self).__init__()


        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if inter_channels == None:
            self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(self.in_channels))
            nn.init.normal_(self.W[0].weight, 0, 0.01)   # todo 尝试设置全为1； 试了效果并不好
            nn.init.zeros_(self.W[0].bias)
            nn.init.normal_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels,kernel_size=1, stride=1, padding=0)

        self.initialize()
    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0,2,1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x )
        f_div_C = F.softmax(f, dim=2)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2,1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W(y)
        z = W_y + x

        return z

    def initialize(self):
        #pass
        for n, m in self.named_children():
            if isinstance(m, nn.Conv2d):
                # xavier(m.weight.data)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.ones_(m.bias)
            elif isinstance(m, nn.Sequential):
                pass
            else:
                RuntimeError('有参数没有初始化')

class ASPP(nn.Module):
    def __init__(self, in_channel=64, depth=64):
        super(ASPP, self).__init__()
        # global average pooling: init nn.AdaptiveAvgPooling2d; also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1, s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1,1) # todo 膨胀率可以是偶数的，
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block5 = nn.Conv2d(in_channel, depth, 3, 1, padding=5, dilation=5)
        self.atrous_block7 = nn.Conv2d(in_channel, depth, 3, 1, padding=7, dilation=7)

        self.conv_1x1_output = nn.Conv2d(depth*5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block3 = self.atrous_block3(x)

        atrous_block5 = self.atrous_block5(x)

        atrous_block7 = self.atrous_block7(x)

        out = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block3,
                                              atrous_block5, atrous_block7], dim=1))
        return out

class NonLocalASPP(nn.Module):
    '''
    non—local -> aspp
    '''
    def __init__(self, in_ch):
        super(NonLocalASPP, self).__init__()
        self.non_local = NonlocalBlock(in_ch)
        self.aspp = ASPP(in_ch)

    def forward(self, x):
        x = self.non_local(x)
        x = self.aspp(x)
        return x

class ASPPNonlocal(nn.Module):
    '''
    apply aspp in non_local
    '''
    def __init__(self, in_channel=64, depth=64):
        super(ASPPNonlocal, self).__init__()
        # global average pooling: init nn.AdaptiveAvgPooling2d; also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1, s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1,1)
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.non_local3 = NonlocalBlock(64)
        self.atrous_block5 = nn.Conv2d(in_channel, depth, 3, 1, padding=5, dilation=5)
        self.non_local5 = NonlocalBlock(64)
        self.atrous_block7 = nn.Conv2d(in_channel, depth, 3, 1, padding=7, dilation=7)
        self.non_local7 = NonlocalBlock(64)

        self.conv_1x1_output = nn.Conv2d(depth*5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block3 = self.atrous_block3(x)
        atrous_block3 = self.non_local3(atrous_block3)

        atrous_block5 = self.atrous_block5(x)
        atrous_block5 = self.non_local5(atrous_block5)

        atrous_block7 = self.atrous_block7(x)
        atrous_block7 = self.non_local7(atrous_block7)

        out = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block3,
                                              atrous_block5, atrous_block7], dim=1))
        return out




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    input = torch.randn(2,4,10,10)
    in1 = input[0,0,:,:]
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(in1)
    # net = NonlocalBlock(4,bn_layer=True)
    # net = NonLocalASPP(4)
    net = ASPPNonlocal(4)
    out = net(input)

    # print(net.W[0].weight)

    out1 = out[0,0,:,:].detach().numpy()
    plt.subplot(1,2,2)
    plt.imshow(out1)
    plt.show()

    print(out.shape)
    # print(in1)
    # print(out1)




