import torch
import random
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import math


def load_data(path, batch_size):
    """
    加载图像
    :param path: 数据集路径
    :param batch_size: 一次加载的张数
    :return:
    """
    train_set = datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]))
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return train_set


def img_grad(x, flag):
    """
    计算x的梯度, 随机取sigma， 计算∂x^2 + ∂y^2的值，若小于sigma则将∂x,∂y置为0, 否则不变
    :param x: [b, ch, height, width]
    :param flag: 1：要对梯度进行处理，0：直接计算出梯度即可
    :return:
    """
    batchsz = x.size()[0]
    channel = x.size()[1]
    h_x = x.size()[2]  # height
    w_x = x.size()[3]  # width

    # 计算梯度∂x, ∂y (-1, 1)
    x_grad_x = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]  # b*c*256*255
    x_grad_y = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]  # b*c*255*256
    x_grad_x = torch.cat((x_grad_x, x[:, :, :, 0:1] - x[:, :, :, w_x-1:w_x]), dim=3)  # b*c*256*256
    x_grad_y = torch.cat((x_grad_y, x[:, :, 0:1, :] - x[:, :, h_x-1:h_x, :]), dim=2)  # b*c*256*256
    # 随机设置阈值sigma为0.001-0.5
    thresh = []
    if flag:
        sigma = torch.zeros((batchsz, 1, 1, 1), device="cuda")
        for i in range(batchsz):
            a = round(random.uniform(0.01, 1.2), 3)
            # a = 0.5
            sigma[i, 0, 0, 0] = a
            thresh.append(a)
        # 计算∂x^2 + ∂y^2, 把各通道求和得到一个单通道图
        # temp_x = torch.sum(torch.pow(x_grad_x, 2), dim=1).unsqueeze(1)
        # temp_y = torch.sum(torch.pow(x_grad_y, 2), dim=1).unsqueeze(1)
        # 计算|∂x| + |∂y|, 把各通道求和得到一个单通道图
        temp_x = torch.sum(torch.abs(x_grad_x), dim=1).unsqueeze(1)
        temp_y = torch.sum(torch.abs(x_grad_y), dim=1).unsqueeze(1)
        # x,y方向求和
        temp = temp_x + temp_y
        # 单通道图中大于sigma的置为1，小于sigma的置为0
        zeros = torch.zeros((batchsz, 1, 1, 1), device="cuda")
        ones = torch.ones((batchsz, 1, 1, 1), device="cuda")
        temp = torch.where(temp < sigma, zeros, ones)
        x_grad_x = torch.mul(x_grad_x, temp)
        x_grad_y = torch.mul(x_grad_y, temp)
    x_grad_x = (x_grad_x + 1) * 0.5
    x_grad_y = (x_grad_y + 1) * 0.5
    return x_grad_x, x_grad_y, thresh


def produce_lamd(batch, h, w):
    """
    产生batch个 h*w的值随机的张量
    :param batch:
    :param h:
    :param w:
    :return:
    """
    lamd_batch = torch.zeros((batch, 1, h, w), device="cuda")

    for i in range(batch):
        # lam = random.uniform(-5,0)
        lam = random.uniform(0.004, 0.8)
        # tmp = torch.full((1, 1, h, w), round(pow(10, lam), 5), device="cuda")
        tmp = torch.full((1, 1, h, w), round(lam, 1), device="cuda")
        lamd_batch[i, 0, ...] = tmp

    return lamd_batch

def produce_sigma(batch, h, w):
    """
    产生batch个 h*w的值随机为1-10的张量
    :param batch:
    :param h:
    :param w:
    :return:
    """
    sigma_batch = torch.zeros((batch, 1, h, w), device="cuda")
    # lams = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # l2_loss

    for i in range(batch):
        # lamd = int(random.random() * 10 + 1)  # 1 - 10  # l1_loss
        # index = int(random.random() * 11)  # l2_loss
        sigma = round(random.uniform(0.1, 1.0), 1)
        tmp = torch.full((1, 1, h, w), sigma, device="cuda")
        sigma_batch[i, 0, ...] = tmp

    return sigma_batch

def compute_grad(x, th):
    """
    MyDataset需要用的一个函数
    :param x: [ch, height, width]
    """
    channel = x.size()[0]
    h_x = x.size()[1]  # height
    w_x = x.size()[2]  # width

    # 计算梯度 取绝对值
    x_grad_x = torch.abs(x[:, :, 1:] - x[:, :, :w_x - 1])  # 128*127
    x_grad_y = torch.abs(x[:, 1:, :] - x[:, :h_x - 1, :])  # 127*128

    # 修改size
    x_grad_x = torch.cat((x_grad_x, torch.zeros(channel, h_x, 1)), dim=2)  # 32*1*128*128
    x_grad_y = torch.cat((x_grad_y, torch.zeros(channel, 1, w_x)), dim=1)  # 32*1*128*128

    if th:
        m = torch.nn.Threshold(th, 0)  # if x < 0.5 => 0
        x_grad_x = m(x_grad_x)
        x_grad_y = m(x_grad_y)
        del m

    return x_grad_x, x_grad_y
