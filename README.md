# U-Net(Convolutional Networks for Biomedical Image Segmentation)

2024 0401

从wzmiaomiao获取代码，尝试学习。

尝试unet 可以运行

但是没有完成mps更改。

没有完成wandb匹配。

没有在kaggle上跑大量实验。

把batch size 从4改到32，风扇也不转，也看不出来apple gpu动了没有。

0402 0041 

arch

去探索能和wandb联动的代码了。

---
## 参考
* https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_segmentation
* [github milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
* [github pytorch/vision](https://github.com/pytorch/vision)
---
## 环境配置：
* Python 3.8
* Pytorch
* 最好使用GPU训练
* 详细环境`requirements.txt`
---
## 文件结构：
```
  ├── src: 搭建U-Net模型代码
  ├── train_utils: 训练、验证以及多GPU训练相关模块
  ├── my_dataset.py: 自定义dataset用于读取DRIVE数据集(视网膜血管分割)
  ├── train.py: 以单GPU为例进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  └── compute_mean_std.py: 统计数据集各通道的均值和标准差
  └── kaggle/working 假设在kaggle运行
```
---
## DRIVE数据集下载地址：
* 官网地址： [https://drive.grand-challenge.org/](https://drive.grand-challenge.org/)
* 百度云链接： [https://pan.baidu.com/s/1Tjkrx2B9FgoJk0KviA-rDw](https://pan.baidu.com/s/1Tjkrx2B9FgoJk0KviA-rDw)  密码: 8no8

下载下来是`DRIVE`文件夹，放入`dataset`文件夹

---
## 实验
* 准备好数据集
  * 在“my_dataset.py”中设置`--data-path`设置为自己存放`DRIVE`文件夹所在的**根目录**
* 若要使用单GPU或者CPU训练，直接使用train.py训练脚本
* 若要使用多GPU训练，使用`torchrun --nproc_per_node=8 train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`(例如我只要使用设备中的第1块和第4块GPU设备)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`
* 运行`train.py`
* `train-mps.py`出现错误`进程已结束，退出代码为 138 (interrupted by signal 10:SIGBUS)`。未解决。
* py：143，炸了
---
## 注意事项
* 在使用预测脚本时，要将`weights_path`设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改`--num-classes`、`--data-path`和`--weights`即可，其他代码尽量不要改动
---
## 使用U-Net在DRIVE数据集上训练得到的权重(仅供测试使用)
- 链接: https://pan.baidu.com/s/1BOqkEpgt1XRqziyc941Hcw  密码: p50a

## 了解 U-Net网络 bilibili
* [https://www.bilibili.com/video/BV1Vq4y127fB/](https://www.bilibili.com/video/BV1Vq4y127fB/)

## U-Net代码的分析
* [https://b23.tv/PCJJmqN](https://b23.tv/PCJJmqN)
---
## 本项目U-Net默认使用双线性插值做为上采样，结构图如下
![u-net](unet.png)
