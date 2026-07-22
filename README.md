# ML2021Spring-HW

李宏毅机器学习 2021 Spring 作业学习记录。这个仓库主要保存我跟着课程完成作业时的 notebook、调参记录和一些学习中的理解。

## Contents

| Homework | Topic | Main Tools |
| --- | --- | --- |
| [HW01](HW01/HW01.ipynb) | COVID-19 cases prediction / regression | PyTorch, NumPy, Matplotlib, Colab |
| [HW02-1](HW02/HW02-1.ipynb) | Phoneme classification | PyTorch, NumPy, Colab GPU |
| [HW03](HW03/HW03.ipynb) | Food-11 image classification / CNN | PyTorch, torchvision, Colab GPU, Kaggle |

## HW01 Overview

HW01 关注从表格数据中预测 COVID-19 相关指标，是一个适合练习机器学习基础流程的回归任务。

学习重点：

- 数据下载与读取
- 自定义 `Dataset` 与 `DataLoader`
- 训练集、验证集与测试集流程
- PyTorch 模型定义、训练和预测
- 损失曲线可视化与基础调参

## HW02-1 Overview

HW02-1 是语音音素分类任务，需要用 TIMIT 资料中的 frame-level acoustic features 预测对应的 phoneme 类别。相比 HW01，这一题更接近一个完整的分类训练流程：输入维度更高，数据量更大，也更明显地暴露出模型容量、正规化和验证集表现之间的关系。

这次主要练习了：

- 用 `Dataset` 和 `DataLoader` 处理较大的 `.npy` 数据
- 使用 `CrossEntropyLoss` 完成多分类训练
- 将隐藏层激活函数从 `Sigmoid` 改为 `ReLU`
- 在全连接层后加入 `BatchNorm1d`
- 用 training / validation accuracy 和 loss 判断模型是否过拟合
- 在 Colab 中使用 GPU 跑较大规模训练


HW02-1 做完之后最大的感受是，深度学习作业里“知道思路”和“真正写出能跑的 PyTorch 代码”之间还有一段距离。模型结构本身并不复杂，但数据切分、label 类型、`DataLoader`、device、loss、accuracy 这些细节只要有一个没对上，训练就会直接报错。

这次也第一次比较直观地看到过拟合：训练集 accuracy 一直上涨、training loss 一直下降，但 validation accuracy 到一定程度后不再提升，validation loss 反而变高。这个现象比单纯看公式更有冲击力，也让我意识到 validation set 的作用不是摆设，而是判断模型有没有真的学到可泛化规律。

目前 HW02-1 还只是一个基础版本，后续可以继续尝试 dropout、weight decay、early stopping、learning rate scheduler 等方法，看看能不能在不过拟合的前提下继续提高 validation accuracy。

## HW03 Overview

HW03 是 Food-11 食物图片分类任务，需要用 CNN 预测图片对应的 11 种食物类别。相比 HW01 和 HW02-1，这一题开始真正处理图像数据：输入不再是整理好的表格或 `.npy` 特征，而是原始图片，因此需要同时关注图片读取、数据增强、卷积网络结构、验证集表现和 Kaggle submission。

这次主要练习了：

- 用 `DatasetFolder` 读取按类别文件夹组织的图片数据
- 区分 training transform 和 validation / testing transform
- 使用 `torchvision.transforms.v2` 做图像数据增强
- 尝试 `RandomResizedCrop`、`ColorJitter`、`RandomRotation` 和 `RandomErasing`
- 用 `Conv2d`、`BatchNorm2d`、`ReLU`、`MaxPool2d` 搭建 CNN baseline
- 使用 `CrossEntropyLoss` 完成 11 类图片分类训练
- 用 training / validation accuracy 和 loss 判断 CNN 是否过拟合
- 初步尝试用 unlabeled data 和 pseudo-label 做半监督学习
- 在 Colab 中处理 `DataLoader`、`num_workers` 和 GPU 训练稳定性问题


HW03 给我的感受更强烈：深度学习不是把模型写出来就结束了，真正麻烦的是让训练流程稳定跑完，并且让 validation accuracy 真的提升。baseline 跑了很久，training accuracy 可以接近 100%，但 validation accuracy 只有 50% 左右，这让我很直观地看到 CNN 很容易记住训练集，而不一定学到能泛化到新图片的特征。

这次也遇到了不少工程细节问题，比如 Colab 里的 `DataLoader` worker 报错、`num_workers` 和训练速度之间的取舍、batch size 对运行稳定性的影响。这些问题看起来不像机器学习理论，但如果处理不好，模型根本跑不起来，或者跑得非常慢。

在尝试 Medium 和 Hard 的过程中，我对 data augmentation 和 semi-supervised learning 的理解也更具体了。random crop、color jitter、random erasing 不是随便让图片变花，而是在逼模型不要依赖某个固定位置、颜色条件或局部特征。pseudo-label 也不是简单地把无标签数据全部加进去，而是要先让模型有一定可靠性，再挑高置信度样本加入训练，否则错误标签会反过来伤害模型。

所以 HW03 最大的收获是：提升模型效果不是单靠训练更久或者模型更大，而是要观察训练集和验证集之间的差距，再有针对性地调整数据增强、正则化、checkpoint 和半监督策略。

目前 HW03 还只是学习过程中的基础版本，后续可以继续尝试 dropout、weight decay、early stopping、learning rate scheduler、best checkpoint 和更稳的 pseudo-label 策略，看看能不能在不过拟合的前提下继续提高 validation accuracy。

## How to Run

推荐直接使用 Colab 打开 notebook：

- [Open HW01 in Colab](https://colab.research.google.com/github/Caspian315/LiHongyiML-2021hw/blob/main/HW01/HW01.ipynb)
- [Open HW02-1 in Colab](https://colab.research.google.com/github/Caspian315/LiHongyiML-2021hw/blob/main/HW02/HW02-1.ipynb)
- [Open HW03 in Colab](https://colab.research.google.com/github/Caspian315/LiHongyiML-2021hw/blob/main/HW03/HW03.ipynb)

也可以在本地 Jupyter 环境运行：

```bash
pip install torch torchvision numpy matplotlib jupyter
jupyter notebook
```

Notebook 中的数据下载依赖课程提供的链接；如果链接失效，可以从原课程/Kaggle 页面手动下载并放到 notebook 工作目录。

## Learning Notes

这个仓库不是完整课程复刻，而是个人学习过程记录。每次作业会尽量保留能复现主要流程的 notebook，同时记录我在实现时真正卡住过的地方。

## Reference

- Course repository: [ML2021-Spring](https://github.com/ga642381/ML2021-Spring)
- HW01 task: COVID-19 Cases Prediction
- HW02-1 task: Phoneme Classification
- HW03 task: Food-11 Image Classification
- Semi-supervised learning slides: [semi (v3)](https://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/semi%20(v3).pdf)
