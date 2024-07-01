# ImageSmoothing

## 项目简介

该项目基于无监督学习实现可调参数的图像保边滤波。可通过调节lambda, sigma的大小来控制图像的保边滤波程度。

## 环境介绍

```python

python==3.6.1
numpy==1.12.1
visdom==0.1.8.9
requests==2.14.2
torchvision==0.4.0
torch==1.2.0
Pillow==8.0.1
PyQt5==5.15.1

```

## 项目结构

```sh
│  .gitignore
│  caculatePsnr.m
│  evaluate.py
│  L0_smoothing.py
│  main.py
│  psnr.py
│  README.md
│  test.py
├─evaluate_set
├─loss_function
│  │  L0_loss.py
│  │  L1_loss.py
│  │  L2_loss.py
│  │  test.py
├─model
│  ├─l1+l1_fcnbn
│  ├─l2+l1_fcnbn
│  ├─l2+l1_noabs
│  ├─l2_fcnbn
│  ├─l2_grad_0_1
│  ├─l2_noabs
│  ├─test
│  └─v0.1
├─net
│  │  autoencoder.py
│  │  cae.py
│  │  fcn.py
│  │  fcnBn.py
│  │  fcnRes.py
├─predict_set
│  ├─L0_smoothing
│  └─L1_L2
├─result
│  ├─compare
│  │  ├─L1
│  │  └─L2
│  ├─fcnbn_L1
│  ├─fcnbn_l2
│
├─test_set256
├─util
│  │  dataset.py
│  │  getImg.py
│  │  gui.py
│  │  imageCutting.py
│  │  sampling.py
│  │  tool.py
│  │  writeFile.py
```

+ model/ :  保存训练好的网络
  
  + l1+l1_fcnbn: 损失函数为L1+L1的形式，网络采用FcnBn
  + l2+l1_fcnbn: 损失函数为L2+L1的形式，网络采用FcnBn
  + l2+l1_noabs: 损失函数为L1+L1的形式，网络采用FcnBn, 相较于前者，对于计算梯度时不做绝对值处理
  + l2_fcnbn: 损失函数为L2+L2的形式，网络采用FcnBn
  + l2_noabs: 损失函数为L2+L2的形式，网络采用FcnBn, 相较于前者，对于计算梯度时不做绝对值处理
  + l2_grad_0_1:损失函数为L2+L2的形式，网络采用FcnBn, 相较于前者，对于计算梯度时不做绝对值处理, 而是将梯度+1、*0.5归一化到（0， 1）
  + v0.1:该文件夹下的模型，网络采用FcnBn，相较于前者，对于梯度的最后一行(列)的值不同，是用第一行(列)-最后一行(列);损失函数改为λ(Sp −Ip)^2 + ((∂xSp − ∂xIp)^2 + (∂ySp − ∂yIp)^2). λ的取值范围是(1e-5, 1), sigma是(0.001, 0.50)
  + v0.2:该文件夹下的模型，网络采用FcnBn，对图像最后一行、一列复制，对于梯度的最后一行(列)的值，是用第一行(列)-最后一行(列);损失函数改为(Sp −Ip)^2 + 1e5*λ((∂xSp − ∂xIp)^2 + (∂ySp − ∂yIp)^2). λ的取值范围是(0, 1), sigma是(0.001, 0.50)
  + v1.0:该文件夹下的模型，网络采用FcnBn，对图像最后一行、一列复制，对于梯度的最后一行(列)的值，是用第一行(列)-最后一行(列);损失函数为(Sp −Ip)^2 + λ(|∂xSp − ∂xIp| + |∂ySp − ∂yIp|). λ的取值范围是(0, 10), sigma是(0.001, 0.50)

+ net/ : 自定义的几种网络结构, 后期基本上只用到了FcnBn。

+ processed_data/ : 保存训练集信息

+ util/ : 保存公用的函数
  + dataset.py : 定义MyDataset， 预处理训练集
  + getImg.py : 从百度上爬取图片用作网络测试
  + tool.py : 保存计算图像梯度、生成lambda的函数
  + writeFile.py : 将训练集信息写入precessed_data/data.txt
  + gui.py : 可视化界面  

+ main.py : 训练网络

+ evaluate.py ： 测试训练好的网络

+ loss_function: 存放自定义损失函数

+ L0_smoothing.py: L0近似方法的实现

## 项目运行

+ 项目训练
  + 使用 PASCALVOC dataset中的17000张图像；将尺寸统一处理为256*256， 其中15000张作为训练集，2000张图像作为验证集。
  
  + 运行： python main.py

+ 项目测试
  + 测试集： 使用从百度上爬取的图片
  + 运行： python evaluate.py cpu/gpu

## 问题记录

1. 发现loss怎么训练都不收敛, 什么学习率，batchsize都调过了, 最后发现是loss函数中输入与输出维度不匹配。要注意的是x[:, 0:1, ...]和x[:, 0, ...]返回的维度并不一样。前者不会改变维度， 后者会减小一个维度。

2. 装cv2的时候报错：OSError: [WinError 17] 系统无法将文件移到不同的磁盘驱动器。: 'd:\\install_software\\anaconda3\\lib\\site-packages\\numpy', 原因是调用了对应的库，文件被占用了，所以系统不能复制文件；简单的方法就是关掉正在使用Python的工具。

3. 与传统方法做对比时发现测试结果会有偏色现象，排查发现损失函数中对于梯度的处理略有问题。原来是梯度的三个通道的像素值只要值小于sigma就置为0， 现改为：|r| + |g| + |b| < simma -> 0。

4. 对梯度图各通道进行处理时，用for循环遍历每个像素时间复杂度太高。进行优化：将三个通道求和得到一个单通道矩阵，后与sigma做对比，大于sigma置1， 小于sigma置0，再用得到的这个矩阵与原梯度图各通道点乘。测试处理一张256&times;256&times;3的图，最终时间从49s -> 0.0009s

5. 网络测试得到的tensor转为PILImage保存时会出现图像失真问题，待解决。保存为png格式即可,不要保存为jpg...

6. L0损失函数没办法训练，只能做L0的近似，在已经训练好的L1，L2的模型上通过迭代不断去近似L0。

7. 在计算梯度时，对梯度取了绝对值后再对其做其他的运算，发现理论上不应该将负梯度置为正，于是改为不取绝对值，但训练效果反而不如之前，正在考虑原因。解决方案：对输入图的梯度值进行sigma处理后，全部值加1再乘以0.5，归一化到（0， 1）的范围，再输入网络中，对于输出图的梯度也如此处理归一化到（0， 1）。

8. 测试L0的时候，发现初始λ=0.01达不到参考[2]中的效果， 查看他的代码，发现他是先对各通道求平方，再对各通道求和。我是先求和再平方，所以值肯定比他的大。

9. 关于偏色问题，考虑依然是梯度处理上的问题，之前的处理方式是：对于行列的最后一行直接补零， 现改为由第一行(列)-最后一行(列)。这样每两个像素值间都会有一个约束关系，理论上能行得通。

10. 训练和测试不要写一起，即使加了model.eval()也不会固定网络参数！！

11. λ之前的范围一直很大，但是对于神经网络而言，特征值域不应该跨度太大，我们的训练中，λ作为入参也就是特征值之一，也应该与sigma，及像素值保持差不多的值域， 所以将λ也进行归一化到(0,1),此时改写loss函数，将λ作为输入输出图像差值的系数。也就是改为：λ(S −I)^2 + ((∂xS − ∂xI')^2 + (∂yS − ∂yI')^2), 其中S是输出, I是输入，∂xI'即经过处理后的梯度值。

## TODO

1. ~~实现L0损失函数并训练；~~

2. ~~L2模型与传统方法的对比；~~

3. ~~解决tensor转为PILImage时出现的图像精度丢失问题；~~

4. ~~将论文Image Smoothing via Unsupervised Learning的代码跑起来，环境搭不起来...~~

5. ~~解决梯度为负值时训练产生的问题~~

6. 用新的梯度处理方法去训练L1, L2

7. 用新训练的L2模型去测试L0的效果

## Reference

[1] Fan, Q., Yang, J., Wipf, D., Chen, B., & Tong, X. (2018). Image smoothing via unsupervised learning. ACM Transactions on Graphics (TOG), 37, 1 - 14.

[2] Li Xu, Cewu Lu, Yi Xu, and Jiaya Jia. 2011. Image smoothing via L0 gradient minimization. ACM Trans. Graph. 30, 6 (December 2011), 1–12.
