
from PIL import Image
from torch.utils.data import Dataset
from util.tool import compute_grad
import torch


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), words[2]))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.root = "./pascal_train_set/JPEGImages/"

    def __getitem__(self, index):
        path, lam, th = self.imgs[index]
        img = Image.open(self.root + path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        lam = torch.full((1, img.size()[1], img.size()[2]), lam)
        img_grad_x, img_grad_y = compute_grad(img, float(th))
        data = torch.cat((img, img_grad_x, img_grad_y, lam), dim=0)

        return data

    def __len__(self):
        return len(self.imgs)
