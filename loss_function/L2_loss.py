"""
@文件 :L2_loss.py
@说明 :λ*(out - in)^2 + ((out_gx - in_gx)^2 + (out_gy - in_gy)^2)
@时间 :2020/11/09 19:21:53
@作者 :唐玲
@版本 :1.0
"""


import torch
import torch.nn as nn
from util.tool import img_grad

class L2Loss(nn.Module):
    """
    自定义loss
    在forward中实现损失函数的计算，返回标量
    """
    def __init__(self):
        super(L2Loss, self).__init__()

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

        # 计算损失的第一部分: (out - in)^2
        loss1 = torch.mean(torch.pow((image - y), 2))

        # 计算损失的第二部分: λ * ((out_gx - in_gx)^2 + (out_gy - in_gy)^2)
        y_grad_x, y_grad_y,_,_= img_grad(y, 0)
        l2 = torch.mul(lambd, (torch.pow((x_grad_x - y_grad_x), 2) + torch.pow((x_grad_y - y_grad_y), 2)))

        loss2 = torch.mean(l2)

        # 合并两部分损失
        return loss1 + loss2
