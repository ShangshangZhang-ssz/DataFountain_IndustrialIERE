# DataFountain_IndustrialIERE工业知识图谱关系抽取

## Overview
**这个项目主要是学习使用gplinker对工业知识图谱关系抽取**

## Approach

使用GlobalPointer模型, ent head tail三个模型预测, 使用旋转矩阵, 融合最后四层特征

bert使用hfl_chinese_roberta_wwm_ext模型, 还可用hfl_chinese_macbert_base等模型, 其他参数较大的模型这里因为显存问题未尝试, 待后续补试

使用模拟退火学习调度器, 搜索最佳阈值, 发现大概threshold负数的时候F1分数最佳, 具体原因不明, 怀疑与模型结构相关比如dropout, 损失函数loss的定义, max_len的大小等等有关系.

max_len为512较佳, 否则ent loss损失过大, 影响整体的loss与Val的F1分数

使用外部数据集

## Environment
**详见requirements.txt**

## Dataset
**数据来自于DataFountain平台**
[工业知识图谱关系抽取-高端装备制造知识图谱自动化构建 竞赛 - DataFountain](https://www.datafountain.cn/competitions/584/datasets)

## Usage
模型需要去Hugging Face下载, 模型下载内容为:

config.json

pytorch_model.bin

tokenizer.json

tokenizer_config.json

vocab.txt

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