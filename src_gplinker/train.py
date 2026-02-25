# 基础库
import os
import sys
import time
import json
import random
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 自然语言处理相关库
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# 深度学习相关库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自定义库
from config import Config
import utils
import data_utils
import model

# 基础环境配置
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', Config["pd_set_option_max_colwidth"])
utils.set_seed(Config["random_seed"])

# 数据准备
rel2id, id2rel = data_utils.build_schema(Config["path_data_train_raw"])
tokenizer = AutoTokenizer.from_pretrained(Config["path_pretrain_model"])

with open(Config["path_data_train_raw"], 'r', encoding='utf-8') as f:
    data_all = [json.loads(line) for line in f if line.strip()]

if Config.get("use_extra_data"):
    with open(Config["path_data_train_other_raw"], 'r', encoding='utf-8') as f:
        extra_data = [json.loads(line) for line in f if line.strip()]
    
    official_count = len(data_all)
    data_all.extend(extra_data)
    print(f"已合并外部数据！总样本量: {len(data_all)} (官方: {official_count} + 外部: {len(extra_data)})")
else:
    print(f"仅使用官方数据训练。总样本量: {len(data_all)}")

train_data, val_data = train_test_split(data_all, test_size=Config["val_size"], random_state=Config["random_seed"])

dataset_train = data_utils.RE_Dataset(train_data, tokenizer, len(rel2id), rel2id, id2rel, is_train=True)
dataset_val = data_utils.RE_Dataset(val_data, tokenizer, len(rel2id), rel2id, id2rel, is_train=True)

dataloader_train = DataLoader(dataset_train, batch_size=Config["batch_size"], shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=Config["batch_size"], shuffle=False)

# 模型初始化与优化器配置
model = model.GPLinkerModel(len(rel2id)).to(Config["device"])

# 分层学习率：实体分支使用更低的学习率，防止震荡
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    # 实体分支参数
    {'params': [p for n, p in param_optimizer if 'entity' in n and not any(nd in n for nd in no_decay)],
     'weight_decay': Config["weight_decay"],
     'lr': Config["learning_rate"] * 0.8},  # 实体分支学习率降低20%
    {'params': [p for n, p in param_optimizer if 'entity' in n and any(nd in n for nd in no_decay)],
     'weight_decay': 0.0,
     'lr': Config["learning_rate"] * 0.8},
    # 其他参数
    {'params': [p for n, p in param_optimizer if 'entity' not in n and not any(nd in n for nd in no_decay)],
     'weight_decay': Config["weight_decay"]},
    {'params': [p for n, p in param_optimizer if 'entity' not in n and any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=Config["learning_rate"])

total_steps = len(dataloader_train) * Config["epochs"]
num_warmup_steps = int(total_steps * Config["warmup_ratio"])

# 调整调度器，更平缓的学习率下降
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps,
    num_cycles=0.2  # 更小的周期，学习率下降更慢
)

# 早停机制
best_val_f1 = 0.0
patience_counter = 0
patience = 3  # 增加patience，给模型更多学习时间
best_global_threshold = 0.0

print("开始训练模型...")

for epoch in range(Config["epochs"]):
    epoch_loss = 0.0
    model.train()
    
    # ========== 移除梯度累积相关代码 ==========
    for step, batch in enumerate(dataloader_train):
        ids, mask, y_ent, y_head, y_tail = [x.to(Config["device"]) for x in batch[:5]]
        p_ent, p_head, p_tail = model(ids, mask)

        # 调整Loss权重，重点关注实体损失
        loss_ent = utils.multilabel_categorical_crossentropy(p_ent, y_ent.unsqueeze(1))
        loss_head = utils.multilabel_categorical_crossentropy(p_head, y_head)
        loss_tail = utils.multilabel_categorical_crossentropy(p_tail, y_tail)

        # 实体损失权重提升至2.0，让模型优先学习实体识别
        loss = (loss_ent * 2.0 + loss_head * 1.0 + loss_tail * 1.0) / 4.0
        
        # 直接反向传播，无梯度累积
        loss.backward()
        epoch_loss += loss.item()

        # 每步都更新参数（移除梯度累积的条件判断）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # 验证阶段
    # 训练集指标
    train_metrics = utils.evaluate(model, dataloader_train, Config["device"], id2rel, threshold=0.0)
    
    # 阈值搜索
    best_f1_this_epoch = 0
    best_threshold_this_epoch = 0
    best_val_metrics = None

    # 更精细的阈值搜索
    for ts in np.arange(-5, 1, 0.5):  # 步长从1.0改为0.5，更精细
        val_metrics = utils.evaluate(model, dataloader_val, Config["device"], id2rel, threshold=ts)
        if val_metrics["f1"] > best_f1_this_epoch:
            best_f1_this_epoch = val_metrics["f1"]
            best_threshold_this_epoch = ts
            best_val_metrics = val_metrics

    # 打印日志
    print(f"--- [Epoch {epoch+1}] ---")
    print(f"--- Train Statistics ---")
    print(f"Loss: {train_metrics['loss']:.4f} (Ent: {train_metrics['ent_loss']:.4f}, Head: {train_metrics['head_loss']:.4f}, Tail: {train_metrics['tail_loss']:.4f})")
    print(f"Train F1: {train_metrics['f1']:.4f} | Train precision: {train_metrics['precision']:.4f} | Train recall: {train_metrics['recall']:.4f}")
    print(f"--- Val Statistics ---")
    print(f"Loss: {best_val_metrics['loss']:.4f} (Ent: {best_val_metrics['ent_loss']:.4f}, Head: {best_val_metrics['head_loss']:.4f}, Tail: {best_val_metrics['tail_loss']:.4f})")
    print(f"Val F1: {best_val_metrics['f1']:.4f} | Val precision: {best_val_metrics['precision']:.4f} | Val recall: {best_val_metrics['recall']:.4f} | Best Threshold: {best_threshold_this_epoch:.1f}")

    # 早停与模型保存
    if best_f1_this_epoch > best_val_f1:
        best_val_f1 = best_f1_this_epoch
        best_global_threshold = best_threshold_this_epoch
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_threshold': best_global_threshold,
        }, Path(Config["path_model_saved"]))
        print(f"验证集 F1 提升至 {best_val_f1:.4f}，最佳阈值已更新为 {best_global_threshold:.1f}！")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"验证集 F1 已连续 {patience} 个 epoch 没有提升，停止训练！")
            break

print(f"训练结束！最佳验证集 F1: {best_val_f1:.4f}，最佳阈值: {best_global_threshold:.1f}")