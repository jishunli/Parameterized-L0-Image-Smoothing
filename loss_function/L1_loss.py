"""
@文件 :L1_loss.py
@说明 :自定义loss函数：(out - in)^2 + lambda(|out_gx - in_gx| + |out_gy - in_gy|)
@时间 :2020/11/06 15:05:53
@作者 :唐玲
@版本 :1.0
"""
import torch
import torch.nn as nn
from util.tool import img_grad


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y):
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

        # y求梯度:
        y_grad_x, y_grad_y, _ = img_grad(y, 0)  # [32, 1, h, w]
        l2 = torch.mul(lambd, (torch.abs(x_grad_y - y_grad_y) + torch.abs(x_grad_x - y_grad_x)))
        loss2 = torch.mean(l2)

        return loss1 + loss2
