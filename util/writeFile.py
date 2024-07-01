"""
将图片， lambda， threshold 写入data.txt, 
后续训练加载data.txt中保存的数据
"""

import os
# import random

dir1 = '../test_set256/train/test'  # 图片文件存放地址
txt1 = '../processed_data/test.txt'  # 图片文件名存放txt文件地址
f1 = open(txt1, 'a')  # 打开文件流
lams = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ths = [0.05, 0.1, 0.15, 0.2]
for filename in os.listdir(dir1):
    for i in range(len(lams)):
        for j in range(len(ths)):
            lam = lams[i]  # 1 - 100
            thresh = ths[j]
            f1.write(filename + " " + str(lam) + " " + str(thresh))  # 写文件
            f1.write("\n")  # 换行
f1.close()  # 关闭文件流
