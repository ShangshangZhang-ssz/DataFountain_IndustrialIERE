import os
import time
import json
import random
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import transformers
from transformers import BertModel, BertTokenizerFast
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset

# 自定义库
from config import Config

class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, heads, head_size=64, RoPE=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size, heads * head_size * 2)

    def sin_cos_position_embedding(self, seq_len, device):
        # 生成旋转位置矩阵
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        indices = torch.arange(0, self.head_size // 2, dtype=torch.float, device=device).unsqueeze(0)
        indices = torch.pow(10000, -2 * indices / self.head_size)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings.reshape(seq_len, self.head_size)

    def forward(self, x, mask):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = self.dense(x)
        x = torch.stack(torch.chunk(x, 2, dim=-1), dim=-1)
        x = x.reshape(batch_size, seq_len, self.heads, self.head_size, 2)
        qw, kw = x[..., 0], x[..., 1]

        # RoPE 位置编码
        if self.RoPE:
            pos_emb = self.sin_cos_position_embedding(seq_len, x.device)
            cos_pos = pos_emb[:, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[:, 0::2].repeat_interleave(2, dim=-1)
            
            def rotate_half(x):
                x1, x2 = x[..., 0::2], x[..., 1::2]
                return torch.stack([-x2, x1], dim=-1).reshape_as(x)
            
            qw = qw * cos_pos[None, :, None, :] + rotate_half(qw) * sin_pos[None, :, None, :]
            kw = kw * cos_pos[None, :, None, :] + rotate_half(kw) * sin_pos[None, :, None, :]

        # 计算 logits
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        
        # 排除填充
        mask = mask.unsqueeze(1).unsqueeze(2)
        logits = logits - (1 - mask) * 1e12
        return logits

class GPLinkerModel(nn.Module):
    def __init__(self, num_rel):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config["path_pretrain_model"], output_hidden_states=True)
        hidden_size = self.bert.config.hidden_size
        
        # ========== 关键修改1：增强实体分支的特征提取 ==========
        # 为实体分支单独设计特征提取层
        self.entity_feature = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.4),  # 更高的dropout防止实体分支过拟合
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # 通用特征精炼层
        self.feature_refiner = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3)
        )
        
        # ========== 关键修改2：实体分支增加额外的GlobalPointer层 ==========
        self.entity_gp1 = GlobalPointer(hidden_size, 1, RoPE=True)
        self.entity_gp2 = GlobalPointer(hidden_size, 1, RoPE=True)  # 双层实体检测
        self.head_gp = GlobalPointer(hidden_size, num_rel, RoPE=True)
        self.tail_gp = GlobalPointer(hidden_size, num_rel, RoPE=True)
        
        # 实体分支的残差连接
        self.entity_residual = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        context = outputs.last_hidden_state 
        context = nn.functional.dropout(context, p=0.1, training=self.training)
        
        # ========== 关键修改3：实体分支单独的特征处理 ==========
        entity_context = self.entity_feature(context)  # 实体分支专用特征
        entity_context = entity_context + self.entity_residual(context)  # 残差连接
        
        # 双层实体检测（融合结果）
        ent_logits1 = self.entity_gp1(entity_context, attention_mask)
        ent_logits2 = self.entity_gp2(entity_context, attention_mask)
        ent_logits = (ent_logits1 + ent_logits2) / 2  # 融合双层结果
        
        # 头/尾分支使用通用特征
        refined_context = self.feature_refiner(context)
        head_logits = self.head_gp(refined_context, attention_mask)
        tail_logits = self.tail_gp(refined_context, attention_mask)
        
        return ent_logits, head_logits, tail_logits