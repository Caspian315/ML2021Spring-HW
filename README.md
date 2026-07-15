# ML2021Spring-HW

李宏毅机器学习 2021 Spring 作业学习记录。当前仓库主要整理 HW01：COVID-19 Cases Prediction，通过 PyTorch 完成一个回归预测任务。

## Contents

| Homework | Topic | Main Tools |
| --- | --- | --- |
| [HW01](HW01/HW01.ipynb) | COVID-19 cases prediction / regression | PyTorch, NumPy, Matplotlib, Colab |

## HW01 Overview

HW01 关注从表格数据中预测 COVID-19 相关指标，是一个适合练习机器学习基础流程的回归任务。

学习重点：

- 数据下载与读取
- 自定义 `Dataset` 与 `DataLoader`
- 训练集、验证集与测试集流程
- PyTorch 模型定义、训练和预测
- 损失曲线可视化与基础调参

## How to Run

推荐直接使用 Colab 打开 notebook：

[Open HW01 in Colab](https://colab.research.google.com/github/Caspian315/ML2021Spring-HW/blob/main/HW01/HW01.ipynb)

也可以在本地 Jupyter 环境运行：

```bash
pip install torch numpy matplotlib jupyter
jupyter notebook HW01/HW01.ipynb
```

Notebook 中的数据下载依赖课程提供的链接；如果链接失效，可以从原课程/Kaggle 页面手动下载并放到 notebook 工作目录。

## Learning Notes

这个仓库不是完整课程复刻，而是个人学习过程记录。后续如果继续补作业，会按 `HWxx/` 目录追加 notebook 和说明。

## Reference

- Course repository: [ML2021-Spring](https://github.com/ga642381/ML2021-Spring)
- HW01 task: COVID-19 Cases Prediction
