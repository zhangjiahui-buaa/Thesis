## 摘要

## 绪论（1～5）

### 研究背景及研究意义

### 研究现状

### 工作内容/目标

### 论文结构

## 相关原理和技术（6～12）

- 图像分类任务
  - 卷积神经网络
  - 视觉transformer
  - 预训练视觉模型
- 文本分类任务

  - self-attention
  - 预训练语言模型
- 多模态分类任务

  - 多模态transformer
  - late/early fusion

## 设计与实现（13～25）

在介绍具体的方法之前，先对一些下文常用到的符号进行定义

|         $X_t$          |             文本输入             |
| :--------------------: | :------------------------------: |
|         $X_i$          |             图像输入             |
| $Y_t(\overline {Y_t})$ | 文本对应的标签（文本模型的预测） |
| $Y_i(\overline {Y_i})$ | 图像对应的标签（图像模型的预测） |
|   $Y(\overline {Y})$   |   综合文本，图像的标签（同上）   |
|         $M_t$          |        用于编码文本的模型        |
|         $M_i$          |        用于编码图像的模型        |
|          $M$           |            多模态模型            |
|                        |                                  |

### 数据预处理

- 文本预处理

  本文选取的数据集中的文本大多数为英语，因此首先将其全部转为小写，在去除收尾的空白符号（换行，制表符），在直接使用空格进行分词。分词之后，利用BertTokenizer进行转化为词表中的索引，并在首部加上【CLS】占位符，尾部加上【SEP】占位符。记为$X_t$

- 图像预处理

  为了适配模型，首先将图像的大小调整为256\*256，在截取图像中间的224\*224个像素点作为最终的图像。由于模型大多数都在Imagenet上预训练，因此需要将输入的图片按照Imagenet数据集的均值和方差进行归一化。具体来说。图像为RGB三通道，故对每一个通道都进行归一化。记为$X_i$
  $$
  X_i \leftarrow \frac{X_i - mean}{std}
  $$
   

### 实现细节

Python + Pytorch + Huggingface + Timm(Pytorch-image-models) + streamlit

AdamW 

Tesla T4

更具体的超参数列在附录中

### 基于视觉transformer的图像情感分类

​	对于图像情感分类，这里采用以Vision Transformer（ViT）为代表的一系列视觉transformer。视觉transformer最初提出是用于图像分类，目标检测这类视觉任务中，即识别出一张图像中的物体，本文将其称为图像的浅层信息。但情感识别任务需要判断出图像中的物体所蕴含的情感，本文将其称为图像的深层信息。

​	具体来说，给定图像输入$X_i$
$$
\overline{L_i} = MLP(ViT(X_i)) \\
\overline{Y_i} = \arg \max(Softmax(\overline{L_i})
$$
​	其中，$MLP$为线性层，用于将ViT编码后的隐变量变换为输出维度的向量，输出维度即为情感种类个数，通常为二或者三。

​	训练过程中用交叉熵作为损失函数
$$
Loss = CrossEntrophy(\overline{L_i}, Y_i)
$$


- 贴图

### 基于BERT的文本情感分类

​	对于文本情感分类任务，本文采用BERT。

​	具体来说，给定文本输入$X_t$
$$
\overline{L_t} = MLP(BERT(X_t)) \\
\overline{Y_t} = \arg \max(Softmax(\overline{L_t})
$$
​	其中，$MLP$为线性层，用于将BERT编码后的隐变量变换为输出维度的向量，输出维度即为情感种类个数，通常为二或者三。

​	训练过程中同样使用交叉熵作为损失函数
$$
Loss = CrossEntrophy(\overline{L_t}, Y_t)
$$




- 贴图

### 多模态情感分类

- 不同模态输入单独编码

  ![avatar](figure1.pdf)

- 不同模态输入统一编码

  ![avatar](figure2.pdf)

- 贴图

### 多模态特征融合（是否放在ablation studies中）

### Ablation studies

- 是否需要预训练
- ensemble models
- Faster R-CNN feature exractor

### 错误分析（随机选100个分析）

### 模型部署

- 前端后端

## 模型展示（26～30）

## 附录

- 超参数
- 数据集细节









