import torch
from torch import nn


class ResBlock(nn.Module):
    """
    构建残差块
    """
    def __init__(self, ch_in, ch_out, padding, dilation=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, (3, 3), stride, padding, dilation),
            nn.ReLU(True),
            nn.Conv2d(ch_in, ch_out, (3, 3), stride, padding, dilation),
            nn.ReLU(True)
        )
        # self.extra = nn.Sequential()
        # if change_dim:
        #     self.extra = nn.Sequential(
        #         nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride)
        #     )

    def forward(self, x):
        out = self.conv(x)
        out = out + x
        return out


class ResFcn(nn.Module):
    """
    构建网络结构
    """
    def __init__(self):
        super(ResFcn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(10, 64, (3, 3), 1, 1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), 2, 1),
            nn.ReLU(True)
        )
        self.layer3 = ResBlock(64, 64, 2, 2)
        self.layer4 = ResBlock(64, 64, 4, 4)
        self.layer5 = ResBlock(64, 64, 8, 8)
        self.layer6 = ResBlock(64, 64, 16, 16)
        self.layer7 = ResBlock(64, 64, 1)
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), 2, 1, 1),
            nn.ReLU(True)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 3, (3, 3), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [b, 3, h, w] => [b, 64, h, w]
        out = self.layer2(self.layer1(x))
        out = self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(out)))))
        out = self.layer9(self.layer8(out))
        return out


if __name__ == "__main__":
    a = torch.randn((1, 10, 256, 256))
    m = ResFcn()
    b = m(a)
    print(b.shape)
