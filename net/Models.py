import torch
from torch import nn
from torchvision import ops
import numpy as np

from ImageDataset import ImageDataSet

feature_channel = 1024


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, padding_mode='zero'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 64
        self.block2 = nn.Sequential(
            ConBlock(input_channel=64, kernel_size=3, channels=[64, 64, 256], strides=1),  # 256
            IdentityBlock(input_channel=256, channels=[64, 64, 256], kernel_size=3),
            IdentityBlock(input_channel=256, channels=[64, 64, 256], kernel_size=3),
            IdentityBlock(input_channel=256, channels=[64, 64, 256], kernel_size=3),
        )

        self.block3 = nn.Sequential(
            ConBlock(input_channel=256, kernel_size=3, channels=[128, 128, 512]),
            IdentityBlock(input_channel=512, channels=[128, 128, 512], kernel_size=3),
            IdentityBlock(input_channel=512, channels=[128, 128, 512], kernel_size=3),
            IdentityBlock(input_channel=512, channels=[128, 128, 512], kernel_size=3),
        )

        self.block4 = nn.Sequential(
            ConBlock(input_channel=512, kernel_size=3, channels=[256, 256, feature_channel]),
            IdentityBlock(input_channel=1024, channels=[256, 256, feature_channel], kernel_size=3),
            IdentityBlock(input_channel=1024, channels=[256, 256, feature_channel], kernel_size=3),
            IdentityBlock(input_channel=1024, channels=[256, 256, feature_channel], kernel_size=3),
            IdentityBlock(input_channel=1024, channels=[256, 256, feature_channel], kernel_size=3),
            IdentityBlock(input_channel=1024, channels=[256, 256, feature_channel], kernel_size=3),
        )

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        return y


class ConBlock(nn.Module):
    def __init__(self, input_channel, kernel_size, channels, strides=2):
        super(ConBlock, self).__init__()

        self.mainNet = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channels[0], kernel_size=1, stride=strides),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=1),
            nn.BatchNorm2d(channels[2])
        )

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channels[2], kernel_size=1, stride=strides),
            nn.BatchNorm2d(channels[2])
        )

        self.finalReLU = nn.ReLU()

    def forward(self, x):
        y1 = self.mainNet(x)
        y2 = self.short_cut(x)
        return self.finalReLU(torch.add(y1, y2))


class IdentityBlock(nn.Module):
    def __init__(self, input_channel, channels, kernel_size):
        super(IdentityBlock, self).__init__()
        assert input_channel == channels[2], ["input_channel != channels[2],残差无法相加"]

        self.mainNet = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channels[0], kernel_size=1),
            nn.BatchNorm2d(num_features=channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=1),
            nn.BatchNorm2d(num_features=channels[2]),
        )

        self.finalReLU = nn.ReLU()

    def forward(self, x):
        y = self.mainNet(x)
        z = torch.add(x, y)
        return self.finalReLU(z)


class RPN(nn.Module):
    def __init__(self, anchor_num):
        super(RPN, self).__init__()
        self.anchor_num = anchor_num
        self.cov3_3 = nn.Sequential(
            nn.Conv2d(in_channels=feature_channel, out_channels=512, padding=1, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.classNet = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=anchor_num, kernel_size=1),
            nn.Sigmoid()
        )
        self.regressNet = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=anchor_num * 4, kernel_size=1),
        )

    def forward(self, x):
        feature = self.cov3_3(x)
        batch = feature.shape[0]
        class_result = self.classNet(feature)
        regress_result = self.regressNet(feature)
        class_result = class_result.view(batch, -1, 1)
        regress_result = regress_result.view(batch, -1, 4)
        return class_result, regress_result


class RoIPooling(nn.Module):
    def __init__(self, pooling_regions=14):
        super(RoIPooling, self).__init__()
        self.pooling_regions = pooling_regions

    # x[0] 为共享特征层  x[1]为roi列表
    def forward(self, x):
        feature = x[0]
        print("feature = {}".format(feature.shape))
        rois = x[1].view(-1, 4)

        print("rois = {}".format(rois.shape))
        samples = ops.roi_pool(input=feature, boxes=rois, output_size=(self.pooling_regions, self.pooling_regions))

        return samples


class Detector(nn.Module):
    def __init__(self, num_class):
        super(Detector, self).__init__()
        self.num_class = num_class
        self.feature_extraction = nn.Sequential(
            ConBlock(input_channel=feature_channel, kernel_size=3, channels=[512, 512, 2048], strides=2),
            IdentityBlock(input_channel=2048, kernel_size=3, channels=[512, 512, 2048]),
            IdentityBlock(input_channel=2048, kernel_size=3, channels=[512, 512, 2048]),
            nn.AvgPool2d(kernel_size=7)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=2048, out_features=num_class + 1),
            nn.Softmax(dim=1)
        )

        self.regress = nn.Sequential(
            # 每一个类别对应一个回归框的偏移量，所以输出是 4 * (num_class-1)，背景没有偏移量
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024,out_features=4 * num_class)

        )

    def forward(self, x):
        print("x.shape = {}".format(x.shape))
        feature = self.feature_extraction(x)
        print("feature.shape = {}".format(feature.shape))
        feature = feature.view(feature.shape[0], -1)
        classify = self.classify(feature)
        regress = self.regress(feature)
        return classify, regress


if __name__ == '__main__':
    # dataset = ImageDataSet(file_name='../train.txt')
    # net = Res50()
    # y = net(dataset[0][0].view(1, dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2], ))
    # print(y.shape)

    x = torch.randn(1, 3, 600, 600)
    net = Res50()
    y = net(x)
    print(y.shape)
