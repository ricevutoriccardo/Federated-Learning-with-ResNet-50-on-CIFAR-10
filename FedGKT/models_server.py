# https://github.com/kuangliu/pytorch-cifar
# https://github.com/FrancescoSaverioZuppichini/ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F

norm_type = "batch_norm"


def Norm(planes, type="batch_norm", num_groups=2):
    if type == "batch_norm":
        return nn.BatchNorm2d(planes)
    if type == "group_norm":
        return nn.GroupNorm(num_groups, planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm(planes, type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = Norm(planes, type=norm_type)

        # if input and output spatial dimensions don't match, as in the paper there are 2 options:
        # - A) identity shortcut with zero padding
        # - B) use 1x1 convolution to increase the channel dimension
        # option B is implemented
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                Norm(planes, type=norm_type)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm(planes, type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = Norm(planes, type=norm_type)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = Norm(self.expansion * planes, type=norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                Norm(self.expansion * planes, type=norm_type)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # changed 3 to 16 in order to receive as input the output of ResNet8
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = Norm(64, type=norm_type)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # average pooling before fully connected layer
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet50(n_type="batch_norm"):
    global norm_type
    norm_type = n_type
    return ResNet(Bottleneck, [3, 4, 6, 3])
