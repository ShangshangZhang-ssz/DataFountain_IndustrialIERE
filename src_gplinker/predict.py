# 基础库
import os
import time
import json
import random
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import rich
from rich.console import Console
from rich.theme import Theme
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

# 自然语言处理相关库
import re
import jieba
from gensim.models import Word2Vec
from gensim import corpora, models

import transformers
from transformers import BertModel, BertTokenizerFast
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# 机器学习相关库
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

# 深度学习相关库
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset

# 自定义库
from config import Config
import utils
import data_utils
import model

checkpoint = torch.load(Path(Config["path_model_saved"]), weights_only=False)
rel2id, id2rel = data_utils.build_schema(Config["path_data_train_raw"])
model = model.GPLinkerModel(len(rel2id)).to(Config["device"])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = BertTokenizerFast.from_pretrained(Config["path_pretrain_model"])
test_data = [json.loads(line) for line in open(Config["path_data_test_raw"], 'r', encoding='utf-8') if line.strip()]
test_ds = data_utils.RE_Dataset(test_data, tokenizer, len(rel2id), rel2id, id2rel, is_train=False)
test_loader = DataLoader(test_ds, batch_size=1) 
results = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        ids, mask, text, idx, offsets, _ = batch
        ids, mask = ids.to(Config["device"]), mask.to(Config["device"])
        
        p_ent, p_head, p_tail = model(ids, mask)
        
        ent_matrix = p_ent[0, 0].cpu().numpy() > checkpoint['best_threshold']
        head_matrix = p_head[0].cpu().numpy() > checkpoint['best_threshold']
        tail_matrix = p_tail[0].cpu().numpy() > checkpoint['best_threshold']
        
        # 形状为 [1, seq_len, 2]，取第 0 个样本
        current_offset = offsets[0].cpu().numpy()
        
        entities = {}
        for s, e in zip(*np.where(ent_matrix)):
            start_char = int(current_offset[s][0])
            end_char = int(current_offset[e][1])
            name = text[0][start_char: end_char]
            if name.strip():
                entities[(s, e)] = {"name": name, "pos": [start_char, end_char]}
        
        spo_list = []
        for rel_id in range(len(rel2id)):
            for sh, oh in zip(*np.where(head_matrix[rel_id])):
                for st, ot in zip(*np.where(tail_matrix[rel_id])):
                    if (sh, st) in entities and (oh, ot) in entities:
                        spo_list.append({
                            "h": entities[(sh, st)], 
                            "t": entities[(oh, ot)], 
                            "relation": id2rel[rel_id]
                        })
        
        results.append({
            "ID": idx[0], 
            "text": text[0], 
            "spo_list": spo_list
        })

# 导出结果
with open(Config["path_submission"], 'w', encoding='utf-8') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"预测完成！结果已保存至 {Config['path_submission']}")