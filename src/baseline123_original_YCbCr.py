
#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.SR_decoder import SR_Decoder


def weight_init(module):
    for n, m in module.named_children():
        try:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
            elif isinstance(m, nn.Sequential):
                weight_init(m)
            elif isinstance(m, (nn.ReLU,nn.PReLU, nn.Unfold, nn.PixelShuffle, nn.Sigmoid, nn.AdaptiveAvgPool2d,nn.Softmax,nn.Dropout2d)):
                pass
            else:
                m.initialize()
        except:
            pass

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

## Shared Residual Encoder
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out2)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/data/iopen/lyf/SaliencyOD_in_RSIs/zhuanli/resnet50-19c8e357.pth'), strict=False)


##PPM: CVPR2017 Pyramid Scene Parsing Network
class PPM(nn.Module): 
    def __init__(self, down_dim):
        super(PPM, self).__init__()
        self.down_conv = self.down_conv = nn.Sequential(nn.Conv2d(2048,down_dim , 3,padding=1),nn.BatchNorm2d(down_dim),
             nn.PReLU())
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.initialize()

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv1_up = F.upsample(conv1, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv2_up = F.upsample(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv3_up = F.upsample(conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv4_up = F.upsample(conv4, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))
    
    def initialize(self):
        weight_init(self)

#TSDD: Transposed Saliency Detection Decoder
class FuseBlock(nn.Module):
    def __init__(self, in_channel1, in_channel2):
        super(FuseBlock, self).__init__()
        self.in_channel1 = in_channel1
        self.in_channel2 = in_channel2
        self.fuse = nn.Conv2d(self.in_channel1 + self.in_channel2, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.down1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.down2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.smaps = nn.Conv2d(64, 1, kernel_size = 3, padding = 1)
        self.initialize()

    def forward(self, x, y):
        out = F.relu(self.bn1(self.fuse(torch.cat((x,y), dim = 1))))
        out1 = F.relu(self.bn2(self.down1(out)))
        out1 = F.relu(self.bn3(self.down2(out1)))
        smaps = self.smaps(out1)
        
        return out, smaps
        
    def initialize(self):
        weight_init(self)

## SRAL: Proposed Super-Resolution Assited Learning framework
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.bkbone  = ResNet()
        self.ppm = PPM(down_dim=256)
        
        self.SRDcoder = SR_Decoder()
    
        self.fuse5 = FuseBlock(in_channel1 = 2048,  in_channel2 = 256)
        self.fuse4 = FuseBlock(in_channel1 = 1024,  in_channel2 = 256)
        self.fuse3 = FuseBlock(in_channel1 = 512,  in_channel2 = 256)
        self.fuse2 = FuseBlock(in_channel1 = 256,  in_channel2 = 256)
        self.fuse1 = FuseBlock(in_channel1 = 64,  in_channel2 = 256)
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(1,3,1),
            torch.nn.BatchNorm2d(3),  
            torch.nn.ReLU(inplace=True)
        )
        self.initialize()

    def forward(self, x_rgb, x_YCbCr):
        s1,s2,s3,s4,s5 = self.bkbone(x_rgb)
        """
        s1: torch.Size([2, 64, 112, 112])
        s2: torch.Size([2, 256, 56, 56])
        s3: torch.Size([2, 512, 28, 28])
        s4: torch.Size([2, 1024, 14, 14])
        s5: torch.Size([2, 2048, 7, 7])
        s6: torch.Size([2, 256, 7, 7])
        """
        s6 = self.ppm(s5)

        out5, smap5 = self.fuse5(s5, s6)

        out4, smap4 = self.fuse4(s4, F.interpolate(out5, size = s4.size()[2:], mode='bilinear',align_corners=True))

        out3, smap3= self.fuse3(s3, F.interpolate(out4, size = s3.size()[2:], mode='bilinear',align_corners=True))

        out2, smap2 = self.fuse2(s2, F.interpolate(out3, size = s2.size()[2:], mode='bilinear',align_corners=True))

        out1, smap1 = self.fuse1(s1, F.interpolate(out2, size = s1.size()[2:], mode='bilinear',align_corners=True))

        ## interpolatation
        smap1 = F.interpolate(smap1, scale_factor=2, mode='bilinear',align_corners=True)
        smap2 = F.interpolate(smap2, scale_factor=4, mode='bilinear',align_corners=True)
        smap3 = F.interpolate(smap3, scale_factor=8, mode='bilinear',align_corners=True)
        smap4 = F.interpolate(smap4, scale_factor=16, mode='bilinear',align_corners=True)
        smap5 = F.interpolate(smap5, scale_factor=32, mode='bilinear',align_corners=True)
        
        pred_sr, sr_fea = self.SRDcoder(s2, s6)
        pred_sr = F.interpolate(x_YCbCr, scale_factor=2, mode='bicubic',align_corners=True) + pred_sr #redisual
        
        if self.training:
            
            pred_sr, sr_fea = self.SRDcoder(s2, s6)

            return (smap1, smap2, smap3, smap4, smap5), pred_sr, self.pointwise(smap1), sr_fea
        else:
            return torch.sigmoid(smap1)
        
    def initialize(self):
        weight_init(self)


