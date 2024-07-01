"""
@文件 :L0_loss.py
@说明 :自定义loss函数：(out - in)^2 + lambda((out_gx - in_gx)^0 + (out_gy - in_gy)^0)
@时间 :2020/11/11 16:08:10
@作者 :唐玲
@版本 :1.0
"""


import torch
import torch.nn as nn
from util.tool import img_grad


class L0Loss(nn.Module):
    def __init__(self):
        super(L0Loss, self).__init__()

    def forward(self, x, y, beta):
        """
        :params x: 输入（原图像，x方向梯度，y方向梯度，lambda）
        :params y: 输出（与原图像维度一致的输出图像）
        """
        # 输入为rgb图
        image = x[:, 0:3, ...]
        x_grad_x = x[:, 3:6, ...]
        x_grad_y = x[:, 6:9, ...]
        lambd = x[:, 9:, ...]
        loss1 = torch.mean(torch.pow((image - y), 2))
        # print(loss1)
        # y求梯度:
        y_grad_x, y_grad_y = img_grad(y, 0)
        # 计算L0

        # (1) if value < min, value = 0; else = 1
        # zeros = torch.zeros_like(y_grad_x)
        # ones = torch.ones_like(y_grad_x)
        # temp_x = torch.where(torch.abs(y_grad_x) < 8*1e-3 , zeros, ones)
        # temp_y = torch.where(torch.abs(y_grad_y) < 8*1e-3, zeros, ones)
        # print(temp_x[0, 0, 0:5, 0:5])

        # (2)(Og-Ig)^2 / ((Og-Ig)^2 + s^2)去近似0次方
        # s = 1e-3
        # p_x = torch.pow((y_grad_x - x_grad_x), 2)
        # p_y = torch.pow((y_grad_y - x_grad_y), 2)
        # p1 = p_x / (s + p_x)
        # p2 = p_y / (s + p_y)
        # l2 = torch.mul(lambd, p1 + p2)

        # （3）用0.1次方去近似0次方
        # l2 = torch.mul(lambd, torch.pow((torch.abs(y_grad_x - x_grad_x)+1e-8), 0.1) + torch.pow((torch.abs(y_grad_y - x_grad_y)+1e-8), 0.1))

        # (3) ∑(S −I)^2 +λC(h,v) +β((∂xS −h)^2 + (∂yS −v)^2)   (h, v -> ∂xI, ∂yI)
        beta = beta * 2
        # 计算C
        # |h| + |v|
        hAndv  = x_grad_x + x_grad_y
        # 将|h| + |v|压成单通道
        hAndv = hAndv[:, 0:1, ...] + hAndv[:, 1:2, ...] + hAndv[:, 2:3, ...]
        # 计算batch中每张图的C(h,v)
        C = torch.nonzero(hAndv[0]).size()[0]
        loss2 = lambd[0] * C
        for i in range(1, x.size()[0]):
            C = torch.nonzero(hAndv[i]).size()[0]
            loss2 += lambd[i] * C
        loss2 = torch.mean(loss2)
        # β((∂xS −h)^2 + (∂yS −v)^2)
        loss3 = torch.mean(beta * (torch.pow((y_grad_x - x_grad_x), 2) + torch.pow((y_grad_y - x_grad_y), 2)))
        loss = loss1 + loss2 + loss3


        # l2 = torch.mul(lambd, temp_x + temp_y)
        # loss2 = torch.mean(l2)
        # print(loss2)

        return loss
