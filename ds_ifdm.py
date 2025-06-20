import antialiased_cnns as acn 
# blur-pool 代替 max pool
import torch
import torch.nn as nn
import torch.nn.functional as F


# Bayar
class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x


# TD-Net输出模块
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        x = self.conv(x)
        return x


class conv_1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_1x1, self).__init__()
        self.layer1 = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(), 
          nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    
    def forward(self,x):
        out = self.layer1(x)
        return out



# ✿✿ヽ(°▽°)ノ✿ TD-Net 网络结构
class TD_Net_First(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(TD_Net_First, self).__init__()
        # 上行采样网络，提取原图的主要特征
        self.bay_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.down_up  = RRU_first_down(n_channels, 32)
        self.down1_up = RRU_down(32, 64)
        self.down2_up = RRU_down(64, 128)
        self.down3_up = RRU_down(128, 256)
        self.down4_up = RRU_down(256, 256)
        # 下行采样网络， 提取高通图像主要特征
        #self.down  = RRU_first_down(n_channels, 32)
        #self.down1 = RRU_down(32, 64)
        #self.down2 = RRU_down(64, 128)
        #self.down3 = RRU_down(128, 256)
        #self.down4 = RRU_down(256, 256)
        # 特征融合层
        self.c =     conv_1x1(512, 256).cuda()
        # 输出网络主模块
        self.up1 = RRU_up(512, 128)
        self.up2 = RRU_up(256, 64)
        self.up3 = RRU_up(128, 32)
        self.up4 = RRU_up(64, 32)
        self.out = outconv(32, n_classes)

    def forward(self, x, y):
        # 原始图像采样网络
        
        x1 = self.down_up(x)
        x2 = self.down1_up(x1)
        x3 = self.down2_up(x2)
        x4 = self.down3_up(x3)
        x5 = self.down4_up(x4)

        # 高通图像采样网络
        y = self.bay_conv(y)
        y1 = self.down_up(y)
        y2 = self.down1_up(y1)
        y3 = self.down2_up(y2)
        y4 = self.down3_up(y3)
        y5 = self.down4_up(y4)
        # 特征融合
        #print(x5.shape)
        #print(y5.shape)
        cmb = torch.cat([x5, y5], dim=1)
        #cmb = torch.cat([x5, y5], dim=1)
        #print(cmb.shape)
        # 短路连接
        x5_new =self.c(cmb)
        #print("x5_new: ", x5_new.shape)
        # 输出网络 
        z = self.up1(x5_new, x4)
        z = self.up2(z, x3)
        z = self.up3(z, x2)
        z = self.up4(z, x1)
        z = self.out(z)
        # activate layers，激活函数
        pred = torch.sigmoid(z)
        # 输出结果
        return pred
    


class RRU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch)
            # Group Normbalization（GN）是一种新的深度学习归一化方式，可以替代BN
        )
        """
        nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))  
        参数：
        in_channel:　输入数据的通道数，例RGB图片通道数为3；
        out_channel: 输出数据的通道数，这个根据模型调整；
        kennel_size: 卷积核大小，可以是int，或tuple；kennel_size=2,意味着卷积大小2， 
                    kennel_size=（2,3），意味着卷积在第一维度大小为2，在第二维度大小为3；
        stride：步长，默认为1，与kennel_size类似，stride=2,意味在所有维度步长为2， 
                stride=（2,3），意味着在第一维度步长为2，意味着在第二维度步长为3；
        padding：　零填充
        """
 
    def forward(self, x):
        x = self.conv(x)
        return x


# RRU-Net主模块
class RRU_first_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        # print("rru_first down")
        super(RRU_first_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
 
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch)
        )
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False)
        )
 
    def forward(self, x):
        # the first ring conv
        # print("forward")
        ft1 = self.conv(x)
        # print("ft1:", ft1)
        r1 = self.relu(ft1 + self.res_conv(x))
        
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + torch.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))
        return r3
 
 
class RRU_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 添加max-blur  layer 
        self.blur = acn.BlurPool(in_ch, stride=2)
 
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))
 
    def forward(self, x):
        x = self.pool(x)
        x = self.blur(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + torch.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))
 
        return r3
 
 
class RRU_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(RRU_up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                nn.GroupNorm(32, in_ch // 2))
 
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
 
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
 
        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
 
        x = self.relu(torch.cat([x2, x1], dim=1))
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + torch.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))
        return r3




if __name__ == "__main__":
    img1 = torch.rand(4, 3, 256, 384).cuda()
    img2 = torch.rand(4, 1, 256, 384).cuda()
    net = TD_Net_First().cuda()
    o = net(img1, img2)
    print(o.shape)
    from torchstat import stat
    import torchsummaryX
    #model=models.resnet18()

    #print(stat(net, [(3, 256, 384),(1,256,384)]))
    torchsummaryX.summary(net, img1,img2)
    
    import thop

    flops, params = thop.profile(net, inputs=(img1, img2))  #input 输入的样本

    print("flops:", flops)