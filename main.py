import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import visdom
from tqdm import tqdm

from net.fcnBn import FcnBn

from util.tool import produce_lamd, load_data, produce_sigma, img_grad
from util.dataset import MyDataset
import math
from torch import nn
import time
from loss_function.L1_loss import L1Loss
from tensorboardX import SummaryWriter




def initial_parameters(model):
    """
    初始化网络参数
    :param model: 网络
    """

    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            size = m.weight.shape
            stdv = math.sqrt(
                12 / (size[1] * size[2] * size[3] + size[0] * size[2] * size[3]))
            print(stdv)
            torch.nn.init.uniform_(m.weight, a=-stdv, b=stdv)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


def main():
    # 获取训练集
    train_set = load_data("./pascal_train_set", 16)
    device = torch.device('cuda')
    model = FcnBn()

    criterion = L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(device)
    # 初始化可视化工具
    writer = SummaryWriter()
    x, _ = iter(train_set).__next__()
    # 绘制网络结构

    height = x.size()[2]
    width = x.size()[3]
    # 网络训练
    # min_loss = 0.02851
    time1 = time.time()
    for epoch in range(1):
        running_loss = 0
        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/1', unit='batch') as pbar:
            for batchidx, (imgs, _) in enumerate(train_set):
                imgs = imgs.to(device)
                batch = imgs.size()[0]
                lam = produce_lamd(batch, height, width)
                grad_x, grad_y, sigma= img_grad(imgs, 1)
                x = torch.cat((imgs, grad_x, grad_y, lam), dim=1)
                x_hat = model(x)
                loss = criterion(x, x_hat)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

        m_loss = running_loss / len(train_set)
        print(epoch, 'train_loss:', m_loss)
        print(epoch, 'lambda:', lam[:, 0, 0, 0])
        print(epoch, 'sigma:', sigma)
        writer.add_scalar('train_loss', m_loss, epoch)
        writer.add_images(tag='train_input', img_tensor=imgs, global_step=epoch)
        writer.add_images(tag='train_output', img_tensor=x_hat, global_step=epoch)
        torch.save(model.state_dict(), "./model/{}_l1_smoothing.pth".format(epoch))
    time2 = time.time()
    print((time2 - time1) / 60 / 60, " 小时")

if __name__ == '__main__':
    main()
