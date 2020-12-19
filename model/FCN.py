import numpy as np
import torch

from torchvision import models
from torch import nn


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight)


class FCN(nn.Module):
    vgg16 = models.vgg16_bn(pretrained=True)

    def __init__(self, num_classes):
        super().__init__()

        # 特征提取的5个模块
        self.stage1 = self.vgg16.features[:7]
        self.stage2 = self.vgg16.features[7:14]
        self.stage3 = self.vgg16.features[14:24]
        self.stage4 = self.vgg16.features[24:34]
        self.stage5 = self.vgg16.features[34:]

        # 两个改变通道数的过渡卷积
        self.conv1 = nn.Conv2d(512, 256, 1)
        self.conv2 = nn.Conv2d(256, num_classes, 1)

        # 反卷积
        self.upsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.upsample_2x_1.weight.data = bilinear_kernel(512, 512, 4)

        self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        self.upsample_2x_2.weight.data = bilinear_kernel(256, 256, 4)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)

        pass

    def forward(self, x):
        '''
            卷积公式：output = ((input + 2 * padding - kernel) / stride) + 1
            池化公式：output = ((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1
            $$L_{out}=floor((L_{in} + 2padding - dilation(kernel_size - 1) - 1)/stride + 1$$
        '''

        # 特征提取
        pool1 = self.stage1(x)      # (bn, 3,   352, 480) -> (bn, 64， 176, 240)
        pool2 = self.stage2(pool1)  # (bn, 64,  176, 240) -> (bn, 128，88,  120)
        pool3 = self.stage3(pool2)  # (bn, 128, 88,  120) -> (bn, 256，44,  60)
        pool4 = self.stage4(pool3)  # (bn, 256, 44,  60)  -> (bn, 512，22,  30)
        pool5 = self.stage5(pool4)  # (bn, 512, 22,  30)  -> (bn, 512，11,  15)

        # pool5 2x上采样得到add1
        pool5_2x = self.upsample_2x_1(pool5)    # (bn, 512，11, 15) -> (bn, 512，22, 30)
        add1 = pool4 + pool5_2x                 # (bn, 512，22, 30) -> (bn, 512，22, 30)

        # add1 2x上采样得到add2
        add1 = self.conv1(add1)                 # (bn, 512, 22, 30) -> (bn, 256，22, 30)
        add1_2x = self.upsample_2x_2(add1)      # (bn, 256，22, 30) -> (bn, 256, 44, 60)
        add2 = pool3 + add1_2x                  # (bn, 256，44, 60) -> (bn, 256，44, 60)

        # add2通过过度卷积改变通道数后 8x上采样得到分数图
        add2 = self.conv2(add2)                 # (bn, 256，44, 60) -> (bn, 12，44, 60)
        y = self.upsample_8x(add2)              # (bn, 12，44, 60) -> (bn, 12，352, 480)

        return y
