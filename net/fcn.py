
import torch
import torch.nn as nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, padding, dilation):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        return out


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, (3, 3), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        # followed 4 blocks
        # [b, 64, h, w] =? [b, 128, h, w]
        self.blk1 = ResBlock(64, 64, 2, 2)
        self.blk2 = ResBlock(64, 64, 4, 4)
        self.blk3 = ResBlock(64, 64, 8, 8)
        self.blk4 = ResBlock(64, 64, 16, 16)
        self.blk5 = ResBlock(64, 64, 1, 1)

        self.deConv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 1, (3, 3), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.blk1(x)
        # x = self.blk1(x)
        x = self.blk2(x)
        # x = self.blk2(x)
        x = self.blk3(x)
        # x = self.blk3(x)
        x = self.blk4(x)
        # x = self.blk4(x)
        x = self.blk5(x)
        x = self.deConv(x)
        x = self.conv3(x)

        return x


def main():
    model = FCN()
    temp = torch.randn(3, 1, 128, 128)
    model(temp)


if __name__ == "__main__":
    main()
