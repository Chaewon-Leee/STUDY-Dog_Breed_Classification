import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
from torchvision import models
from torchvision.models import ResNet152_Weights

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()

        cfg = {
                'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
              }

        self.features = self._make_layers(cfg['VGG16'])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, 120) # 120개의 클래스 분류
        )

    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)

        out = self.classifier(out)
        return F.log_softmax(out, dim=1)

    def _make_layers(self, cfg):
        layers=  []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else :
                layers += [
                          nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=(3,3), stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)
                           ]
                in_channels = x
        return nn.Sequential(*layers)

class resnet(nn.Module):
    def __init__(self):
        super(resnet,self).__init__()

        self.resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        self.linear_layers = nn.Linear(1000, 120)

    def forward(self,x):
        x = self.resnet(x)
        out = self.linear_layers(x)
        self.freezing()
        return F.log_softmax(out, dim=1)

    def freezing(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.linear_layers.parameters():
            param.requires_grad = True

## MobileNet

class MobileNet(nn.Module):
    def __init__(self, ch_in, num_classes):
        super(MobileNet,self).__init__()

        cfg = { # 1번째 레이어 + dw
            'MobileNet': [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024, 'P'],
            'stride': [2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]
            }

        self.ch_in = 3
        self.num_classes = 120

        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(1024, self.num_classes), # fully connected layer
            # nn.ReLU(True),
            # # nn.Dropout(),
            # nn.Linear(1000, self.num_classes) # classification
        )

    def dw_pw(self, in_channels, out_channels, stride):
            return [nn.Sequential(
                    # dw
                    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    # pw
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                    )]

    def forward(self,x):
        out = self.features(x)
        # out = out.view(x.size(0), -1)
        out = out.view(-1, 1024) # 1024로 맞춰주기
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)

    def _make_layers(self, cfg):
        layers=  []
        in_channels = self.ch_in
        for idx, x in enumerate(cfg['MobileNet']):
            s = cfg['stride'][idx]
            if idx != 0 and x != 'P':
                layers += self.dw_pw(in_channels, x, s)
            elif idx == 0:
                layers += [nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, bias=False, stride=s, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                    )]
            elif x == 'P': # AvgPool
                layers += [
                    nn.AvgPool2d(7, stride=s)
                ]
            in_channels = x
        return nn.Sequential(*layers)


class MobileNetV1(nn.Module):
    def __init__(self, ch_in):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, 120)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# def weight_standardization(weight: torch.Tensor, eps: float):
#     c_out, c_in, *kernel_shape = weight.shape
#     weight = weight.view(c_out, -1)
#     var, mean = torch.var_mean(weight, dim=1, keepdim=True)
#     weight = (weight - mean) / (torch.sqrt(var + eps))
#     return weight.view(c_out, c_in, *kernel_shape)

# class VGG_WS(nn.Module):
#     def __init__(self):
#         super(VGG_WS,self).__init__()

#         self.cfg = {
#                 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#               }

#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 1000),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(1000, 120) # 120개의 클래스 분류
#         )



#     def forward(self,x):
#         in_channels = 3
#         for c in self.cfg:
#             if c == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else :
#                 # WS
#                 weight_standardization(weight=self.weight, eps=1e-5)
#                 # convolutional layer
#                 F.conv2d(x, self.weight, self.bias, stride=1,
#                         padding=1, dilation=1, groups=1)
#                 # BatchNorm
#                 BatchNorm = nn.BatchNorm2d(c), # c : 입력받는 데이터의 channels
#                 BatchNorm(x)
#                 # Relu
#                 R = nn.ReLU(inplace=True)
#                 R(x)
#             in_channels = c
#         out = out.view(out.size(0),-1)
#         out = self.classifier(out)
#         return F.log_softmax(out, dim=1)
