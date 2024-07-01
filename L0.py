"""
@文件 :L0_smoothing.py
@说明 :L0的近似方法 ∑p (Sp −Ip)^2 +λC(h,v) +β((∂xSp −hp)^2 + (∂ySp −vp)^2)
@时间 :2020/11/22 14:41:52
@作者 :唐玲
@版本 :1.0
"""
from net.fcnBn import FcnBn
import torch
from PIL import Image
from torchvision import transforms
import time
import os
import torch.nn.functional as F



def iteration(lam, beta, s):
    """
    @description: 用新的s, beta计算新的h、v
    ---------
    @param: 
    -------
    @Returns: 新的h, v, c
    -------
    """
    batchsz = s.size()[0]
    channel = s.size()[1]
    height = s.size()[2]
    width = s.size()[3]
    # 计算S的梯度
    dx = s[:, :, :, 1:] - s[:, :, :, :width - 1]  # 256 * 255
    dy = s[:, :, 1:, :] - s[:, :, :height - 1, :]  # 255 * 256
    dx = torch.cat((dx,  s[:, :, :, 0:1] - s[:, :, :, width-1:width]), dim=3)  # 32*1*256*256
    dy = torch.cat((dy,  s[:, :, 0:1, :] - s[:, :, height-1:height, :]), dim=2)  # 32*1*256*256
    # 计算h, v
    # 计算|∂x| + |∂y|, 把各通道求和得到一个单通道图
    temp_x = torch.sum(torch.abs(dx), dim=1).unsqueeze(1)
    temp_y = torch.sum(torch.abs(dy), dim=1).unsqueeze(1)
    temp = temp_x + temp_y
    zeros = torch.zeros((batchsz, 1, 1, 1), device="cuda")
    ones = torch.ones((batchsz, 1, 1, 1), device="cuda")
    temp = torch.where(temp < lam/beta, zeros, ones)
    dx = dx * temp
    dy = dy * temp
    dx = (dx + 1) * 0.5
    dy = (dy + 1) * 0.5
    return dx, dy


def handleImage(img, height, width):
    # 把图像尺寸改为双数
    signal = 0
    if height % 2 == 1 and width % 2 == 1:
        img = F.interpolate(img, size=[height+1, width+1])
        height += 1
        width += 1
        signal = 1
    elif height % 2 == 1:
        img = F.interpolate(img, size=[height+1, width])
        height += 1
        signal = 2
    elif width % 2 == 1:
        img = F.interpolate(img, size=[height, width+1])
        width += 1
        signal = 3
    return img, height, width, signal


def saveImage(device, img, fn, path):
    """
    @description: transform tensor to image and save to local
    ---------
    @param: img:tensor, out: output path
    -------
    @Returns: 1/0
    -------
    """
    if device == torch.device('cuda'): img = img.cpu()
    if not os.path.exists(path):
        os.makedirs(path)
    new_img_PIL = transforms.ToPILImage()(img[0])
    new_img_PIL.save(path + "/" + fn)


def L0():
    """
    @description: 
    ---------
    @param: 
    -------
    @Returns: 
    -------
    """
    transform = transforms.Compose([

        transforms.ToTensor()
    ])
    device = torch.device('cuda')
    # 加载网络参数
    model = FcnBn()
    modelPath = './model/v1.0/l1_smoothing.pth'
    model.load_state_dict(torch.load(modelPath))
    model = model.to(device)
    # 图像预处理
    dir1 = "./Data/"
    filenames = os.listdir(dir1)
    timeall = 0
    seconds = []
    # print(time.time())

    # beta的增长因子
    factor = 2
    # 一次L0迭代的总次数
    interationTimes = 8
    # lambda
    valueL = 0.02
    # beta初始值
    valueB = 1 * valueL
    batch = 1
    height = 720
    width = 1280
    # 记录开始时间

    for fn in filenames:
        if fn == '.DS_Store':
            continue
        # 图片名称
        fullfilename = dir1 + "3.jpg"
        img = Image.open(fullfilename).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        
        # 记录原始图像
        img1 = img
        # 一次L0中当前迭代次数
        currentIrTime = 0

        # 测试
        time_1 = time.time();
        with torch.no_grad():
            model.eval()
            beta = valueB

            while currentIrTime < interationTimes:
                lam = torch.full((batch, 1, height, width), beta, device='cuda')
                grad_x, grad_y = iteration(valueL, beta, img)
                x = torch.cat((img1, grad_x, grad_y, lam), dim=1)
                img = model(x)
                beta *= factor
                currentIrTime += 1
        cost = time.time() - time_1
        saveImage(device, img, fn, './result')

    #测速
    #print(cost)

if __name__ == "__main__":
    L0()