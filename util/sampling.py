"""
@文件 :sampling.py
@说明 :本文件实现两个函数，一个用于将图像下采样为几张图像；另一个将几张图像合并为一张图像。
@时间 :2020/11/03 20:48:25
@作者 :唐玲
@版本 :1.0
"""

import torch
import cv2
from PIL import Image


def down_sampling(image, size):
    """
    @description: 将图像下采样为几张图像
    ---------
    @param: image: tensor,要进行采样的图像, size:下采样的大小[height, width]
    -------
    @Returns: 采样后的图像组成的一个batch
    -------
    """
    dim = image.dim()
    if dim == 4:
        image = image.squeeze(0)
    height = image.size(1)
    width = image.size(2)
    # 计算图像与采样后的图像的宽高比
    h_num = int(height / size[0])
    w_num = int(width / size[1])
    res = torch.empty((h_num * w_num, image.size(0), size[0], size[1]))
    # 隔点采样
    for h in range(h_num):
        for w in range(w_num):
            res[h * w_num + w] = image[:, h:height:h_num, w:width:w_num]

    return res, h_num, w_num


def up_sampling(image, size):
    """
    @description: 将多张图像合并为一张
    ---------
    @param: image: 一个batch的图像[b,c,h,w], size:[h_num, w_num]图像宽高分别需要多少张图进行合并
    -------
    @Returns: 合并后的图像
    -------
    """
    height = image.size(2)*size[0]
    width = image.size(3)*size[1]
    res = torch.empty((1, image.size(1), height, width))

    for h in range(size[0]):
        for w in range(size[1]):
            res[0, :, h:height:size[0], w:width:size[1]] = image[h * size[1] + w]
    return res


def GaussianBlur(source):
    img = cv2.GaussianBlur(source, (3, 3), 10)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    return pilimg


if __name__ == "__main__":
    a = torch.arange(1, 33).view(1, 1, 4, -1)
    print(a)
    b, h, w = down_sampling(a, [4, 2])
    print(b)
    d = up_sampling(b, [h, w])
    print(d)