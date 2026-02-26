# DataFountain_IndustrialIERE工业知识图谱关系抽取

## Overview
**这个项目主要是学习使用gplinker对工业知识图谱关系抽取**

## Approach

### 数据分析

1. 分析官方train.json与外部数据train_other.json与测试数据evalA.json的文本与token的长度分布

   发现max_len设置为512的时候数据覆盖较佳, 但仍存在长尾分布, 这对模型实体识别造成影响, 可使用数据滑动窗口或数据切分处理, 但由于时间问题并未尝试.

   并且外部数据大多较短, max_len为256的时候数据覆盖为98%, evalA数据与train数据整体相似

2. 查看train.json与train_other.json的relation分布, 发现部件故障占90%, 其余3种情况较少, 数据分布极度不均衡, 模型recall较低, 可使用过采样进行数据增强, 与文本增强, 但由于时间问题并未尝试.

### 数据处理

1. 只将外部数据进行提取成官方数据样本格式

### 模型结构

1. 使用预训练bert模型, 采用旋转编码增强模型的位置信息能力, 使用GlobalPointer, 模型采用4个独立模型分别预测实体 $\times$ 2, head, tail, 并对实体模型分支做了独立特征提取层与残渣连接, head与tail模型进行norm与dropout.取最后一层hidden, 双层实体融合结果.

### 训练方法

1. 采用分层学习率与模拟退火学习调度器, 采用AdamW为优化器
2. split20%为验证数据, 采用早停机制, 每次寻找最佳logits的阈值, 并保存最佳阈值与模型

### 后处理

1. 未进行任何后处理

### 训练日志

**详见src/experiment.md**

### TOP方案

https://zhuanlan.zhihu.com/p/640074513

https://zhuanlan.zhihu.com/p/639710921

https://zhuanlan.zhihu.com/p/639568647

https://zhuanlan.zhihu.com/p/639330922

https://zhuanlan.zhihu.com/p/639330922

## Environment
详见requirements.txt

## Dataset
**数据来自于DataFountain平台**
[工业知识图谱关系抽取-高端装备制造知识图谱自动化构建 竞赛 - DataFountain](https://www.datafountain.cn/competitions/584/datasets)

## Usage
1. 模型需要去Hugging Face下载, 模型下载内容为:

   config.json

   pytorch_model.bin

   tokenizer.json

   tokenizer_config.json

   vocab.txt

2. **Github展示代码并非最佳, 最新模型代码日志均在src/experiment.md**

## Project Structure
```text
.
├── data/                           # 数据集目录
├── src_gplinker/                   # GPLinker 模型实现与结果
│   ├── model_saved/                # 训练得到的模型权重 (已忽略具体权重文件)
│   ├── submission/                 # 模型提交结果文件夹
│   ├── config.py                   # 配置文件
│   ├── data_utils.py               # 数据处理工具
│   ├── model.py                    # 模型定义文件
│   ├── predict.py                  # 推理/预测脚本
│   ├── readme.md                   # 该模块的说明文档
│   ├── train.py                    # 训练脚本
│   └── utils.py                    # 通用工具函数
├── .gitignore                      # Git 忽略规则配置
├── eda.ipynb                       # 数据探索性分析 Notebook
├── LICENSE                         # 项目许可证文件
├── README.md                       # 项目根目录说明文件
└── requirements.txt                # Python 依赖库列表
```