import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.batchnorm import SynchronizedBatchNorm2d as SBatchNorm

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

class _Residual_Block(nn.Module): 
    def __init__(self, ch):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
            
        self.initialize()
    def forward(self, x): 
        identity_data = x
        output = self.conv2(self.relu(self.conv1(x)))
        output = output + self.residual(identity_data)

        return output 
        
    def initialize(self):
        weight_init(self)

class SR_Decoder(nn.Module):
    def __init__(self):
        
        super(SR_Decoder, self).__init__()
        
        self.down = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            SBatchNorm(64),
            nn.ReLU()
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1, bias=False),
            SBatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            SBatchNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 64, kernel_size=1, stride=1)
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2), #[B, 64, 56, 56] --> [B, 64, 112, 112]
            _Residual_Block(64), #Num = 1
            nn.ConvTranspose2d(64, 32, 2, stride=2), #[B, 64, 112, 112] --> [B, 32, 224, 224]
            _Residual_Block(32), #Num = 1
            nn.ConvTranspose2d(32, 16, 2, stride=2), #[B, 16, 224, 224] --> [B, 64, 224, 224]
            _Residual_Block(16), #Num = 1
        )
        self.pred_sr = nn.Conv2d(16,3,1)
        self.sr_fea = nn.Conv2d(16,3,1)  

        self._init_weight()


    def forward(self, s2, s6):
        """
        s2: first stage feature of ResNet50 [B, 256, 56, 56]
        s6: feature map of PPM [B, 256, 7, 7]
        """
        s2 = self.down(s2)

        s6 = F.interpolate(s6, size=s2.size()[2:], mode='bilinear', align_corners=True)
        f = self.fuse(torch.cat((s2, s6), dim=1))
        f = self.up(f)
        pred_sr = self.pred_sr(f)
        sr_fea = self.sr_fea(f)
        return pred_sr, sr_fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
