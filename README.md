## Overview

此仓库包含我对 Udacity 的 Capstone 项目的 [**Deep Knowledge Tracing**](https://github.com/chrispiech/DeepKnowledgeTracing) 实现。

## 目的

构建并训练LSTM网络，以预测学生正确回答他尚未看到的问题的概率。使用的是 [**ASSISTments Skill-builder data 2009-2010**](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) 公共数据集。

## 结果

这是通过不断修改网络配置，获取的最佳验证损失。

| Test Data (%) | AUC |
| --- | --- |
| 20% | 0,85 |

可以在“Log”文件夹中找到每次尝试的结果，配置和模型权重。

## 要求

您需要Python 3.x x64才能运行这些项目。

如果您还没有安装Python，建议您安装Python 的 [Anaconda](https://www.anaconda.com/download/) 发行版，它几乎包含这些项目中所需的所有软件包。

您也可以从[这里](https://www.python.org/downloads/)安装Python 3.x x64

## 说明

1. 克隆存储库并导航到下载的文件夹。
```	
git clone https://github.com/lccasagrande/Deep-Knowledge-Tracing.git
cd Deep-Knowledge-Tracing
```

2. 安装所需的包：
	- 如果已安装TensorFlow，请键入：
	```
	pip install -e .
	```
	- 如果要使用TensorFlow-GPU进行安装，请按照[本指南](https://www.tensorflow.org/install/) 检查系统上必需的NVIDIA软件。然后键入：
	```
	pip install -e .[tf_gpu]
	```
	- 如果要使用Tensorflow-CPU进行安装，请键入：
	```
	pip install -e .[tf]
	```

3. 导航到src文件夹并打开 notebook.
```	
cd src
jupyter notebook DKT.ipynb
```

