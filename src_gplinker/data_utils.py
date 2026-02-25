import os
import time
import json
import random
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader, Subset

# 自定义库
from config import Config

def build_schema(path):
    rel_set = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            for spo in item.get('spo_list', []):
                rel_set.add(spo['relation'])
    rel2id = {rel: i for i, rel in enumerate(sorted(list(rel_set)))}
    id2rel = {i: rel for rel, i in rel2id.items()}
    return rel2id, id2rel

def get_weighted_sampler(dataset_train, train_data, rel2id):
    # 1. 判定每个样本的“稀有程度”
    sample_weights = []
    
    # 统计全局频率
    rel_counts = {r: 0 for r in rel2id.keys()}
    for item in train_data:
        for spo in item.get('spo_list', []):
            rel_counts[spo['relation']] += 1
            
    for item in train_data:
        # 如果这个样本包含“稀有”类别（比如检测工具），给它极高的权重
        current_rels = [spo['relation'] for spo in item.get('spo_list', [])]
        
        if not current_rels: # 负样本
            weight = 1.0
        else:
            # 权重 = 该样本中所有关系对应频率倒数的最大值
            # 意味着只要包含一个稀有类别，整个样本就被视为稀有
            weight = max([1.0 / (rel_counts[r] + 1) for r in current_rels])
        
        sample_weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True # 允许重复采样
    )
    return sampler

class RE_Dataset(Dataset):
    def __init__(self, data, tokenizer, num_rel, rel2id, id2rel, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.num_rel = num_rel
        self.rel2id = rel2id
        self.id2rel = id2rel
        self.is_train = is_train

    def search(self, pattern, sequence, pos):
        n = len(pattern)
        candidate = [i for i in range(len(sequence)) if sequence[i:i + n] == pattern]
        if not candidate: return -1
        a = []
        for i in candidate:
            s = ''.join(self.tokenizer.decode(sequence[1:i]).split(' '))
            a.append([abs(len(s) - pos[0]), i])
        return sorted(a, key=lambda x: x[0])[0][1]

    def __len__(self): return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item['text']
        encoding = self.tokenizer(text, max_length=Config["max_len"], truncation=True, 
                                 padding='max_length', return_offsets_mapping=True)
        
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        
        # 核心修正：将 offset_mapping 显式转为 Tensor [seq_len, 2]
        # 这样 DataLoader 就能将其堆叠为 [batch_size, seq_len, 2]
        offset_mapping = torch.tensor(encoding['offset_mapping'])
        
        spo_list = item.get('spo_list', [])
        
        if not self.is_train:
            return input_ids, attention_mask, text, item.get('ID', ''), offset_mapping, str(spo_list)

        # 训练标签构建逻辑保持不变...
        entity_labels = np.zeros((Config["max_len"], Config["max_len"]))
        head_labels = np.zeros((self.num_rel, Config["max_len"], Config["max_len"]))
        tail_labels = np.zeros((self.num_rel, Config["max_len"], Config["max_len"]))

        for spo in spo_list:
            s_ids = self.tokenizer.encode(spo['h']['name'], add_special_tokens=False)
            o_ids = self.tokenizer.encode(spo['t']['name'], add_special_tokens=False)
            sh = self.search(s_ids, encoding['input_ids'], spo['h']['pos'])
            oh = self.search(o_ids, encoding['input_ids'], spo['t']['pos'])
            
            if sh != -1 and oh != -1:
                st, ot = sh + len(s_ids) - 1, oh + len(o_ids) - 1
                if st < Config["max_len"] and ot < Config["max_len"]:
                    p_id = self.rel2id[spo['relation']]
                    entity_labels[sh, st] = 1 
                    entity_labels[oh, ot] = 1
                    head_labels[p_id, sh, oh] = 1 
                    tail_labels[p_id, st, ot] = 1 

        return input_ids, attention_mask, \
               torch.tensor(entity_labels, dtype=torch.float), \
               torch.tensor(head_labels, dtype=torch.float), \
               torch.tensor(tail_labels, dtype=torch.float), \
               text, str(spo_list), offset_mapping