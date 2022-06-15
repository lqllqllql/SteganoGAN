# -*- coding: utf-8 -*-

import torch
from torch import nn

# 建立基础模型
class BasicEncoder(nn.Module):
    """
    The BasicEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    
    该模块采用封面图像和数据张量结合，将其转换为隐写图像。

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = False

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    
    def _build_models(self):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            
            self._conv2d(self.hidden_size, 3),
            nn.Tanh(),
        )
        return self.features, self.layers

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        # 将旧的预训练模型转换成新代码版本适用的模型
        if not hasattr(self, 'version'):
            self.version = '1'
    
    # 前向传播
    def forward(self, image, data):
        x = self._models[0](image)
        # 将图像像素展开成一个列表
        x_list = [x]

        for layer in self._models[1:]:
            # torch.cat()：连接给定维度中给定的张量序列。在这里是串联的维度为2维。
            x = layer(torch.cat(x_list + [data], dim=1))
            x_list.append(x)

        if self.add_image:
            x = image + x
        return x

# 残差块模型
class ResidualEncoder(BasicEncoder):
    """
    The ResidualEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = True

    def _build_models(self):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            
            self._conv2d(self.hidden_size, 3),
        )
        return self.features, self.layers


# 全连接层模型    
class DenseEncoder(BasicEncoder):
    """
    The DenseEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = True

    def _build_models(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4
