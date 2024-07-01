# Parameterized-L0-Image-Smoothing

## Introduction

This project is based on unsupervised learning and aims to implement an edge-preserving image filtering method with adjustable parameters. By adjusting the values of `lambda` and `sigma`, the degree of edge preservation in the filtered image can be controlled.

## Environment Configuration

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

## Project Structure

```sh
project/
├── Data/
│   └── input_images/        # Directory for input images
├── loss_function/
│   ├── L0.py/
│   ├── L1.py/
│   └── L2.py/
├── model/                  # Directory for model files (if applicable)
│   └── l1_smoothing.pth.pth    # Trained model file
├── net/                 
│   ├── fcn.py/
│   ├── fcnBn.py/           # Train using this network
│   └── fcnRes.py/
├── result/
│   └── output_images/        # Directory for output images
├─util/                       #tools
│  ├── dataset.py
│  ├── getImg.py
│  ├── gui.py
│  ├── imageCutting.py
│  ├── sampling.py
│  ├── tool.py
│  └── writeFile.py                   
├── L0.py               # Running this function will directly yield the result
├── main.py            # This function is used for training the model
└── README.md            # Project documentation and instructions
```



## Project execution

+ Project training
  + Use 17,000 images from the PASCAL VOC dataset; resize them to a uniform size of 256x256. Out of these, 15,000 images are used for training, and 2,000 images are used for validation.
  
  + Run: python main.py

+ Project testing
  + Run: python L0.py



## Notes 

When running `L0.py`, please ensure that the images you want to process are placed in the `Data` folder. The results of the execution can be found in the `result` folder. It is important to note that for each image, you need to modify the `height` and `width` parameters in the code according to the size of the image. Failure to do so may result in errors!

## Reference

[1] Fan, Q., Yang, J., Wipf, D., Chen, B., & Tong, X. (2018). Image smoothing via unsupervised learning. ACM Transactions on Graphics (TOG), 37, 1 - 14.

[2] Li Xu, Cewu Lu, Yi Xu, and Jiaya Jia. 2011. Image smoothing via L0 gradient minimization. ACM Trans. Graph. 30, 6 (December 2011), 1–12.
