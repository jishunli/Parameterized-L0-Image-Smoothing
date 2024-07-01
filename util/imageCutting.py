"""
@文件 :imageCutting.py
@说明 :本文件用于网络测试时对输入图像进行预处理(图像切割与合并)
@时间 :2020/11/04 10:44:08
@作者 :唐玲
@版本 :1.0
"""

import torch


def image_cutting(image, size):
    """
    @description: 将image切割为多张尺寸为size的图
    ---------
    @param: image：输入图像，size：想要的尺寸
    -------
    @Returns: 切割后的图像
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
    k = 0
    for h in range(0, height, size[0]):
        for w in range(0, width, size[1]):
            res[k] = image[:, h:h+size[0], w:w+size[1]]
            k += 1
    return res

def img_merging(image, size):
    """
    @description: 将多张图像合并为一张
    ---------
    @param: image：一个batch的图像，size：高和宽由几张图像组成
    -------
    @Returns: 合并后的图像
    -------
    """
    height = image.size(2)
    width = image.size(3)
    res = torch.empty((1, image.size(1), height*size[0], width*size[1]))
    k = 0
    for h in range(0, height*size[0], height):
        for w in range(0, width*size[1], width):
            res[0, :, h : h + height, w : w + width] = image[k]
            k += 1
    return res
