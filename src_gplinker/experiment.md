# idea

1. 模型选取最后四层拼接, stack, 或者两者都做后再拼接
2. 数据文本增强, evalb伪标签

# exp01_cv0.5955_lb0.6445

## **approach**

使用GlobalPointer模型, ent head tail三个模型预测, 使用旋转矩阵, 融合最后四层特征

bert使用hfl_chinese_roberta_wwm_ext模型

使用模拟退火学习调度器, 搜索最佳阈值, 发现大概threshold负数的时候F1分数最佳, 具体原因不明, 怀疑与模型结构相关比如dropout, 损失函数loss的定义, max_len的大小等等有关系.

max_len为512较佳, 否则ent loss损失过大, 影响整体的loss与Val的F1分数

使用外部数据集

## config

```python
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Config = {
    # 基本参数
    "console_theme":{
        "info": "cyan",
        "title": "bold italic red",
        "train": "bold blue",
        "eval": "bold green",
        "warn": "bold yellow",
    },
    "pd_set_option_max_colwidth": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # 模型参数
    "epochs": 50,
    "n_splits": 5,
    "batch_size": 12,
    "max_len": 512,
    "val_size": 0.2,
    "random_seed": 42,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "use_extra_data": True,

    # 文件参数
    "path_pretrain_model" :  "C:/Users/ShangshangZhang/Desktop/DLC/Pretrain_model/hfl_chinese_roberta_wwm_ext",
    "path_data_train_raw" : str(PROJECT_ROOT / "data/raw/train.json"),
    "path_data_train_other_raw" : str(PROJECT_ROOT / "data/raw/train_other.json"),
    "path_data_test_raw" : str(PROJECT_ROOT / "data/raw/evalA.json"),
    
    "path_submission" : str(PROJECT_ROOT / "src_gplinker/submission" / "sub_exp01.json"),
    "path_model_saved": str(PROJECT_ROOT / "src_gplinker/model_saved" / "model_exp01"),
}
```

## utils

```python
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
from torch.utils.data import Dataset, DataLoader, Subset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class FGM():
    """
    定义对抗训练 FGM 类
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        """
        对 Embedding 层注入扰动
        :param epsilon: 扰动权重
        :param emb_name: 需要注入扰动的 Embedding 层名称
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 备份当前参数
                self.backup[name] = param.data.clone()
                # 计算梯度范数
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 计算扰动并叠加
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """
        恢复被扰动之前的参数
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    y_pred: [..., num_classes]
    y_true: [..., num_classes] (0 or 1)
    """
    # 这一步能保证不管你是 2 维还是 4 维，逻辑都能闭环
    shape = y_pred.shape
    y_pred = y_pred.reshape(-1, shape[-1])
    y_true = y_true.reshape(-1, shape[-1])
    
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    # 返回每个样本的 loss 之和，再取均值
    return (torch.logsumexp(y_pred_neg, dim=-1) + torch.logsumexp(y_pred_pos, dim=-1)).mean()

def evaluate(model, data_loader, device, id2rel, threshold=0.0):
    model.eval()
    X, Y, Z = 1e-10, 1e-10, 1e-10
    total_loss, total_ent_loss, total_head_loss, total_tail_loss = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in data_loader:
            ids, mask, y_ent, y_head, y_tail, texts, raw_spos, offsets = batch
            ids, mask, y_ent, y_head, y_tail = [x.to(device) for x in [ids, mask, y_ent, y_head, y_tail]]
            
            p_ent, p_head, p_tail = model(ids, mask)
            
            # 计算细分 Loss
            l_ent = multilabel_categorical_crossentropy(p_ent, y_ent.unsqueeze(1))
            l_head = multilabel_categorical_crossentropy(p_head, y_head)
            l_tail = multilabel_categorical_crossentropy(p_tail, y_tail)
            
            batch_loss = (l_ent + l_head + l_tail) / 3
            total_ent_loss += l_ent.item()
            total_head_loss += l_head.item()
            total_tail_loss += l_tail.item()
            total_loss += batch_loss.item()

            for i in range(len(texts)):
                target = eval(raw_spos[i])
                target_set = set()
                for s in target:
                    target_set.add((s['h']['name'], tuple(s['h']['pos']), s['t']['name'], tuple(s['t']['pos']), s['relation']))
                
                # 💡 使用传入的阈值进行硬判定
                ent_matrix = p_ent[i, 0].cpu().numpy() > threshold
                head_matrix = p_head[i].cpu().numpy() > threshold
                tail_matrix = p_tail[i].cpu().numpy() > threshold
                current_offset = offsets[i].cpu().numpy()
                
                entities = {}
                for s, e in zip(*np.where(ent_matrix)):
                    start_char, end_char = int(current_offset[s][0]), int(current_offset[e][1])
                    name = texts[i][start_char: end_char]
                    if name.strip(): entities[(s, e)] = (name, [start_char, end_char])
                
                pred_set = set()
                # 遍历所有已识别出的实体对，检查它们是否存在指定的某种关系
                for (sh, eh), sub_info in entities.items():
                    for (so, eo), obj_info in entities.items():
                        # 遍历每一种关系类型
                        for rel_id in range(len(id2rel)):
                            # 如果 head_matrix 标记了主客体的起始点，且 tail_matrix 标记了主客体的结束点
                            if head_matrix[rel_id, sh, so] and tail_matrix[rel_id, eh, eo]:
                                pred_set.add((
                                    sub_info[0],        # 主体名
                                    tuple(sub_info[1]), # 主体位置 [start, end]
                                    obj_info[0],        # 客体名
                                    tuple(obj_info[1]), # 客体位置 [start, end]
                                    id2rel[rel_id]      # 关系类型
                                ))
                
                # 💡 核心计数逻辑
                X += len(pred_set & target_set) # 预测对的 (TP)
                Y += len(pred_set)              # 预测出的总量 (TP + FP)
                Z += len(target_set)             # 样本真实总量 (TP + FN)

    num_batches = len(data_loader)
    
    # 💡 计算最终指标
    precision = X / Y
    recall = X / Z
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "loss": total_loss / num_batches,
        "ent_loss": total_ent_loss / num_batches,
        "head_loss": total_head_loss / num_batches,
        "tail_loss": total_tail_loss / num_batches,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    return metrics
```

## data_utils

```python
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
```

## model

```python
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

        # 💡 核心修改：注入 RoPE
        if self.RoPE:
            pos_emb = self.sin_cos_position_embedding(seq_len, x.device)
            # 计算旋转后的 q 和 k
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
        self.bert = BertModel.from_pretrained(Config["path_pretrain_model"], output_hidden_states=True) # 开启 hidden_states
        hidden_size = self.bert.config.hidden_size
        
        # 任务解耦：为实体识别增加一个独立的小分支
        self.ent_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(), # 增加非线性，帮助捕捉边界
            nn.Dropout(0.2)
        )
        
        self.entity_gp = GlobalPointer(hidden_size, 1, RoPE=True)
        self.head_gp = GlobalPointer(hidden_size, num_rel, RoPE=True)
        self.tail_gp = GlobalPointer(hidden_size, num_rel, RoPE=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        
        # 核心改进：融合最后 4 层
        # outputs.hidden_states 包含 Embedding 层 + 12 个 Transformer 层
        all_layers = torch.stack(outputs.hidden_states[-4:], dim=0) 
        context = all_layers.mean(dim=0) # [batch, seq, hidden]
        
        # 实体识别使用独立分支特征
        ent_context = self.ent_gate(context)
        
        return self.entity_gp(ent_context, attention_mask), \
               self.head_gp(context, attention_mask), \
               self.tail_gp(context, attention_mask)
```

## train

```python
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

# 基础环境配置
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', Config["pd_set_option_max_colwidth"])
console = Console(theme=Theme(Config["console_theme"]), color_system="truecolor")
utils.set_seed(Config["random_seed"])

# 数据准备
rel2id, id2rel = data_utils.build_schema(Config["path_data_train_raw"])
tokenizer = BertTokenizerFast.from_pretrained(Config["path_pretrain_model"])

with open(Config["path_data_train_raw"], 'r', encoding='utf-8') as f:
    data_all = [json.loads(line) for line in f if line.strip()]

if Config.get("use_extra_data"):
    with open(Config["path_data_train_other_raw"], 'r', encoding='utf-8') as f:
        extra_data = [json.loads(line) for line in f if line.strip()]
    
    official_count = len(data_all)
    data_all.extend(extra_data)
    
    console.print(
        f"已合并外部数据！总样本量: [bold]{len(data_all)}[/bold] "
        f"(官方: {official_count} + 外部: {len(extra_data)})", 
        style="info"
    )
else:
    console.print(f"仅使用官方数据训练。总样本量: {len(data_all)}", style="info")

train_data, val_data = train_test_split(data_all, test_size=Config["val_size"], random_state=Config["random_seed"])

dataset_train = data_utils.RE_Dataset(train_data, tokenizer, len(rel2id), rel2id, id2rel, is_train=True)
dataset_val = data_utils.RE_Dataset(val_data, tokenizer, len(rel2id), rel2id, id2rel, is_train=True)

sampler = data_utils.get_weighted_sampler(dataset_train, train_data, rel2id)

dataloader_train = DataLoader(dataset_train, batch_size=Config["batch_size"], shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=Config["batch_size"], shuffle=False)

# 模型训练
model = model.GPLinkerModel(len(rel2id)).to(Config["device"])
optimizer = torch.optim.AdamW(model.parameters(), lr=Config["learning_rate"], weight_decay=Config["weight_decay"])

total_steps = len(dataloader_train) * Config["epochs"]
num_warmup_steps = int(total_steps * Config["warmup_ratio"])

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,   # 热身步数，通常 0.05~0.1 总步数
    num_training_steps=total_steps,      # 总训练步数
    num_cycles=0.5                        # 半个余弦周期，默认 0.5
)

fgm = utils.FGM(model)

# 早停机制
best_val_f1 = 0.0          # 用于记录最佳 F1
patience_counter = 0       # 早停计数器
patience = 3               # 如果连续 5 个 epoch 验证集没有提升，则停止
best_global_threshold = 0.0       # 最佳evaluate阈值

console.print("开始训练模型...", style="title")
model.train()
for epoch in range(Config["epochs"]):
    for batch in dataloader_train:
        ids, mask, y_ent, y_head, y_tail = [x.to(Config["device"]) for x in batch[:5]]
        p_ent, p_head, p_tail = model(ids, mask)

        # --- 修改后的加权 Loss ---
        loss_ent = utils.multilabel_categorical_crossentropy(p_ent, y_ent.unsqueeze(1))
        loss_head = utils.multilabel_categorical_crossentropy(p_head, y_head)
        loss_tail = utils.multilabel_categorical_crossentropy(p_tail, y_tail)

        loss = (loss_ent * 0.5 + loss_head * 1.0 + loss_tail * 1.0) / 2.5
        loss.backward()

        # 对抗训练
        fgm.attack() # 注入扰动
        p_ent_adv, p_head_adv, p_tail_adv = model(ids, mask)
        loss_ent_adv = utils.multilabel_categorical_crossentropy(p_ent_adv, y_ent.unsqueeze(1))
        loss_head_adv = utils.multilabel_categorical_crossentropy(p_head_adv, y_head)
        loss_tail_adv = utils.multilabel_categorical_crossentropy(p_tail_adv, y_tail)

        loss_adv = (loss_ent_adv * 0.5 + loss_head_adv + loss_tail_adv) / 2.5
        loss_adv.backward() # 再次回传对抗梯度
        fgm.restore() # 恢复原始参数

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # --- 1. 获取 Train 指标 ---    
    # 训练集通常很大，建议 threshold 固定为 0 以节省时间
    train_metrics = utils.evaluate(model, dataloader_train, Config["device"], id2rel, threshold=0.0)

    # --- 2. 阈值搜索与获取 Val 指标 ---
    best_f1_this_epoch = 0
    best_threshold_this_epoch = 0
    best_val_metrics = None

    for ts in np.arange(-5, 0.1, 1.0): 
        val_metrics = utils.evaluate(model, dataloader_val, Config["device"], id2rel, threshold=ts)
        if val_metrics["f1"] > best_f1_this_epoch:
            best_f1_this_epoch = val_metrics["f1"]
            best_threshold_this_epoch = ts
            best_val_metrics = val_metrics

    # --- 3. 打印详细日志 ---
    # 打印 Train 信息
    console.print(f"--- [Epoch {epoch+1}] ---", style="title")
    console.print(f"--- Train Statistics ---", style="train")
    console.print(f"Loss: {train_metrics['loss']:.4f} (Ent: {train_metrics['ent_loss']:.4f}, Head: {train_metrics['head_loss']:.4f}, Tail: {train_metrics['tail_loss']:.4f})", style="train")
    console.print(f"Train F1: {train_metrics['f1']:.4f} | Train precision: {train_metrics['precision']:.4f} | Train recall: {train_metrics['recall']:.4f}", style="train")

    # 打印 Val 信息
    console.print(f"--- Val Statistics ---", style="train")
    console.print(f"Loss: {best_val_metrics['loss']:.4f} (Ent: {best_val_metrics['ent_loss']:.4f}, Head: {best_val_metrics['head_loss']:.4f}, Tail: {best_val_metrics['tail_loss']:.4f})", style="train")
    console.print(f"Val F1: {best_val_metrics['f1']:.4f} | Val precision: {best_val_metrics['precision']:.4f} | Val recall: {best_val_metrics['recall']:.4f} | Best Threshold: {best_threshold_this_epoch:.1f}", style="train")

    # 早停与模型保存
    if best_f1_this_epoch > best_val_f1 or epoch == 0:
        best_val_f1 = best_f1_this_epoch
        best_global_threshold = best_threshold_this_epoch # 更新最佳阈值
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_threshold': best_global_threshold,
        }, Path(Config["path_model_saved"]))
        console.print(f"验证集 F1 提升至 {best_val_f1:.4f}，最佳阈值已更新为 {best_global_threshold:.1f}！", style="warn")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            console.print(f"验证集 F1 已连续 {patience} 个 epoch 没有提升，停止训练！", style="warn")
            break
```

## predict

```python
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
```

## log

```python
开始训练模型...
--- [Epoch 1] ---
--- Train Statistics ---
Loss: 0.0275 (Ent: 0.0549, Head: 0.0139, Tail: 0.0136)
Train F1: 0.0000 | Train precision: 1.0000 | Train recall: 0.0000
--- Val Statistics ---
Loss: 0.0297 (Ent: 0.0596, Head: 0.0149, Tail: 0.0146)
Val F1: 0.0000 | Val precision: 1.0000 | Val recall: 0.0000 | Best Threshold: -3.0
验证集 F1 提升至 0.0000，最佳阈值已更新为 -3.0！
--- [Epoch 2] ---
--- Train Statistics ---
Loss: 0.0109 (Ent: 0.0244, Head: 0.0042, Tail: 0.0039)
Train F1: 0.0018 | Train precision: 1.0000 | Train recall: 0.0009
--- Val Statistics ---
Loss: 0.0119 (Ent: 0.0266, Head: 0.0047, Tail: 0.0044)
Val F1: 0.2315 | Val precision: 0.1615 | Val recall: 0.4084 | Best Threshold: -3.0
验证集 F1 提升至 0.2315，最佳阈值已更新为 -3.0！
--- [Epoch 3] ---
--- Train Statistics ---
Loss: 0.0080 (Ent: 0.0182, Head: 0.0030, Tail: 0.0027)
Train F1: 0.0783 | Train precision: 0.9281 | Train recall: 0.0408
--- Val Statistics ---
Loss: 0.0090 (Ent: 0.0203, Head: 0.0035, Tail: 0.0031)
Val F1: 0.4096 | Val precision: 0.4674 | Val recall: 0.3646 | Best Threshold: -2.0
验证集 F1 提升至 0.4096，最佳阈值已更新为 -2.0！
--- [Epoch 4] ---
--- Train Statistics ---
Loss: 0.0063 (Ent: 0.0145, Head: 0.0024, Tail: 0.0021)
Train F1: 0.3205 | Train precision: 0.8442 | Train recall: 0.1978
--- Val Statistics ---
Loss: 0.0075 (Ent: 0.0169, Head: 0.0029, Tail: 0.0026)
Val F1: 0.4768 | Val precision: 0.4142 | Val recall: 0.5616 | Best Threshold: -2.0
验证集 F1 提升至 0.4768，最佳阈值已更新为 -2.0！
--- [Epoch 5] ---
--- Train Statistics ---
Loss: 0.0052 (Ent: 0.0119, Head: 0.0021, Tail: 0.0018)
Train F1: 0.4617 | Train precision: 0.8490 | Train recall: 0.3170
--- Val Statistics ---
Loss: 0.0067 (Ent: 0.0150, Head: 0.0027, Tail: 0.0023)
Val F1: 0.5162 | Val precision: 0.6109 | Val recall: 0.4469 | Best Threshold: -1.0
验证集 F1 提升至 0.5162，最佳阈值已更新为 -1.0！
--- [Epoch 6] ---
--- Train Statistics ---
Loss: 0.0043 (Ent: 0.0097, Head: 0.0017, Tail: 0.0016)
Train F1: 0.5622 | Train precision: 0.8841 | Train recall: 0.4122
--- Val Statistics ---
Loss: 0.0062 (Ent: 0.0142, Head: 0.0024, Tail: 0.0021)
Val F1: 0.5538 | Val precision: 0.6428 | Val recall: 0.4864 | Best Threshold: -1.0
验证集 F1 提升至 0.5538，最佳阈值已更新为 -1.0！
--- [Epoch 7] ---
--- Train Statistics ---
Loss: 0.0037 (Ent: 0.0083, Head: 0.0015, Tail: 0.0013)
Train F1: 0.5862 | Train precision: 0.9343 | Train recall: 0.4271
Loss: 0.0061 (Ent: 0.0138, Head: 0.0024, Tail: 0.0021)
Val F1: 0.5724 | Val precision: 0.5211 | Val recall: 0.6349 | Best Threshold: -2.0
验证集 F1 提升至 0.5724，最佳阈值已更新为 -2.0！
--- [Epoch 8] ---
--- Train Statistics ---
Loss: 0.0037 (Ent: 0.0083, Head: 0.0015, Tail: 0.0013)
Train F1: 0.5187 | Train precision: 0.9673 | Train recall: 0.3543
--- Val Statistics ---
Loss: 0.0065 (Ent: 0.0149, Head: 0.0026, Tail: 0.0021)
Val F1: 0.5955 | Val precision: 0.6302 | Val recall: 0.5645 | Best Threshold: -2.0
验证集 F1 提升至 0.5955，最佳阈值已更新为 -2.0！
--- [Epoch 9] ---
--- Train Statistics ---
Loss: 0.0028 (Ent: 0.0063, Head: 0.0011, Tail: 0.0011)
Train F1: 0.6945 | Train precision: 0.9557 | Train recall: 0.5454
--- Val Statistics ---
Loss: 0.0060 (Ent: 0.0137, Head: 0.0024, Tail: 0.0021)
Val F1: 0.5813 | Val precision: 0.5486 | Val recall: 0.6183 | Best Threshold: -2.0
```

# exp02_cv0.5944_lb0.6506

## approch

在exp01的基础上, 进行了如下优化.

get_weighted_sampler, 发现关系样本分布特别不均衡, 检测工具”和“组成”等类别样本极少, 使用重采样方法加强少样本数据.

## config

```python
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Config = {
    # 基本参数
    "console_theme":{
        "info": "cyan",
        "title": "bold italic red",
        "train": "bold blue",
        "eval": "bold green",
        "warn": "bold yellow",
    },
    "pd_set_option_max_colwidth": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # 模型参数
    "epochs": 50,
    "n_splits": 5,
    "batch_size": 12,
    "max_len": 512,
    "val_size": 0.2,
    "random_seed": 42,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "use_extra_data": True,

    # 文件参数
    "path_pretrain_model" :  "C:/Users/ShangshangZhang/Desktop/DLC/Pretrain_model/hfl_chinese_roberta_wwm_ext",
    "path_data_train_raw" : str(PROJECT_ROOT / "data/raw/train.json"),
    "path_data_train_other_raw" : str(PROJECT_ROOT / "data/raw/train_other.json"),
    "path_data_test_raw" : str(PROJECT_ROOT / "data/raw/evalA.json"),
    
    "path_submission" : str(PROJECT_ROOT / "src_gplinker/submission" / "sub_exp01.json"),
    "path_model_saved": str(PROJECT_ROOT / "src_gplinker/model_saved" / "model_exp01"),
}
```

## utils

```python
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
from torch.utils.data import Dataset, DataLoader, Subset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class FGM():
    """
    定义对抗训练 FGM 类
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        """
        对 Embedding 层注入扰动
        :param epsilon: 扰动权重
        :param emb_name: 需要注入扰动的 Embedding 层名称
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 备份当前参数
                self.backup[name] = param.data.clone()
                # 计算梯度范数
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 计算扰动并叠加
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """
        恢复被扰动之前的参数
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    y_pred: [..., num_classes]
    y_true: [..., num_classes] (0 or 1)
    """
    # 这一步能保证不管你是 2 维还是 4 维，逻辑都能闭环
    shape = y_pred.shape
    y_pred = y_pred.reshape(-1, shape[-1])
    y_true = y_true.reshape(-1, shape[-1])
    
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    # 返回每个样本的 loss 之和，再取均值
    return (torch.logsumexp(y_pred_neg, dim=-1) + torch.logsumexp(y_pred_pos, dim=-1)).mean()

def evaluate(model, data_loader, device, id2rel, threshold=0.0):
    model.eval()
    X, Y, Z = 1e-10, 1e-10, 1e-10
    total_loss, total_ent_loss, total_head_loss, total_tail_loss = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in data_loader:
            ids, mask, y_ent, y_head, y_tail, texts, raw_spos, offsets = batch
            ids, mask, y_ent, y_head, y_tail = [x.to(device) for x in [ids, mask, y_ent, y_head, y_tail]]
            
            p_ent, p_head, p_tail = model(ids, mask)
            
            # 计算细分 Loss
            l_ent = multilabel_categorical_crossentropy(p_ent, y_ent.unsqueeze(1))
            l_head = multilabel_categorical_crossentropy(p_head, y_head)
            l_tail = multilabel_categorical_crossentropy(p_tail, y_tail)
            
            batch_loss = (l_ent + l_head + l_tail) / 3
            total_ent_loss += l_ent.item()
            total_head_loss += l_head.item()
            total_tail_loss += l_tail.item()
            total_loss += batch_loss.item()

            for i in range(len(texts)):
                target = eval(raw_spos[i])
                target_set = set()
                for s in target:
                    target_set.add((s['h']['name'], tuple(s['h']['pos']), s['t']['name'], tuple(s['t']['pos']), s['relation']))
                
                # 💡 使用传入的阈值进行硬判定
                ent_matrix = p_ent[i, 0].cpu().numpy() > threshold
                head_matrix = p_head[i].cpu().numpy() > threshold
                tail_matrix = p_tail[i].cpu().numpy() > threshold
                current_offset = offsets[i].cpu().numpy()
                
                entities = {}
                for s, e in zip(*np.where(ent_matrix)):
                    start_char, end_char = int(current_offset[s][0]), int(current_offset[e][1])
                    name = texts[i][start_char: end_char]
                    if name.strip(): entities[(s, e)] = (name, [start_char, end_char])
                
                pred_set = set()
                # 遍历所有已识别出的实体对，检查它们是否存在指定的某种关系
                for (sh, eh), sub_info in entities.items():
                    for (so, eo), obj_info in entities.items():
                        # 遍历每一种关系类型
                        for rel_id in range(len(id2rel)):
                            # 如果 head_matrix 标记了主客体的起始点，且 tail_matrix 标记了主客体的结束点
                            if head_matrix[rel_id, sh, so] and tail_matrix[rel_id, eh, eo]:
                                pred_set.add((
                                    sub_info[0],        # 主体名
                                    tuple(sub_info[1]), # 主体位置 [start, end]
                                    obj_info[0],        # 客体名
                                    tuple(obj_info[1]), # 客体位置 [start, end]
                                    id2rel[rel_id]      # 关系类型
                                ))
                
                # 💡 核心计数逻辑
                X += len(pred_set & target_set) # 预测对的 (TP)
                Y += len(pred_set)              # 预测出的总量 (TP + FP)
                Z += len(target_set)             # 样本真实总量 (TP + FN)

    num_batches = len(data_loader)
    
    # 💡 计算最终指标
    precision = X / Y
    recall = X / Z
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "loss": total_loss / num_batches,
        "ent_loss": total_ent_loss / num_batches,
        "head_loss": total_head_loss / num_batches,
        "tail_loss": total_tail_loss / num_batches,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    return metrics
```

## data_utils

```python
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
```

## model

```python
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
```

## train

```python
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
```

## predict

```python
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
```

## log

```python
开始训练模型...
--- [Epoch 1] ---
--- Train Statistics ---
Loss: 0.0190 (Ent: 0.0324, Head: 0.0123, Tail: 0.0121)
Train F1: 0.0000 | Train precision: 1.0000 | Train recall: 0.0000
--- Val Statistics ---
Loss: 0.0208 (Ent: 0.0352, Head: 0.0138, Tail: 0.0135)
Val F1: 0.0000 | Val precision: 1.0000 | Val recall: 0.0000 | Best Threshold: -5.0
验证集 F1 提升至 0.0000，最佳阈值已更新为 -5.0！
--- [Epoch 2] ---
--- Train Statistics ---
Loss: 0.0101 (Ent: 0.0198, Head: 0.0058, Tail: 0.0048)
Train F1: 0.0427 | Train precision: 0.6929 | Train recall: 0.0220
--- Val Statistics ---
Loss: 0.0114 (Ent: 0.0219, Head: 0.0068, Tail: 0.0055)
Val F1: 0.2973 | Val precision: 0.3289 | Val recall: 0.2713 | Best Threshold: -3.5
验证集 F1 提升至 0.2973，最佳阈值已更新为 -3.5！
--- [Epoch 3] ---
--- Train Statistics ---
Loss: 0.0078 (Ent: 0.0167, Head: 0.0035, Tail: 0.0033)
Train F1: 0.4192 | Train precision: 0.5976 | Train recall: 0.3228
--- Val Statistics ---
Loss: 0.0091 (Ent: 0.0191, Head: 0.0043, Tail: 0.0040)
Val F1: 0.4351 | Val precision: 0.4834 | Val recall: 0.3955 | Best Threshold: -1.0
验证集 F1 提升至 0.4351，最佳阈值已更新为 -1.0！
--- [Epoch 4] ---
--- Train Statistics ---
Loss: 0.0060 (Ent: 0.0121, Head: 0.0031, Tail: 0.0028)
Train F1: 0.4359 | Train precision: 0.8048 | Train recall: 0.2989
--- Val Statistics ---
Loss: 0.0081 (Ent: 0.0162, Head: 0.0043, Tail: 0.0037)
Val F1: 0.4956 | Val precision: 0.5475 | Val recall: 0.4526 | Best Threshold: -2.0
验证集 F1 提升至 0.4956，最佳阈值已更新为 -2.0！
--- [Epoch 5] ---
--- Train Statistics ---
Loss: 0.0052 (Ent: 0.0106, Head: 0.0027, Tail: 0.0024)
Train F1: 0.4813 | Train precision: 0.8681 | Train recall: 0.3330
--- Val Statistics ---
Loss: 0.0078 (Ent: 0.0160, Head: 0.0041, Tail: 0.0034)
Val F1: 0.5398 | Val precision: 0.5654 | Val recall: 0.5164 | Best Threshold: -2.5
验证集 F1 提升至 0.5398，最佳阈值已更新为 -2.5！
--- [Epoch 6] ---
--- Train Statistics ---
Loss: 0.0044 (Ent: 0.0091, Head: 0.0021, Tail: 0.0020)
Train F1: 0.6451 | Train precision: 0.7510 | Train recall: 0.5654
--- Val Statistics ---
Loss: 0.0073 (Ent: 0.0152, Head: 0.0036, Tail: 0.0031)
Val F1: 0.5547 | Val precision: 0.6331 | Val recall: 0.4936 | Best Threshold: -0.5
验证集 F1 提升至 0.5547，最佳阈值已更新为 -0.5！
--- [Epoch 7] ---
--- Train Statistics ---
Loss: 0.0035 (Ent: 0.0070, Head: 0.0019, Tail: 0.0016)
Train F1: 0.6833 | Train precision: 0.8290 | Train recall: 0.5812
--- Val Statistics ---
Loss: 0.0074 (Ent: 0.0153, Head: 0.0037, Tail: 0.0031)
Val F1: 0.5818 | Val precision: 0.6024 | Val recall: 0.5626 | Best Threshold: -1.5
验证集 F1 提升至 0.5818，最佳阈值已更新为 -1.5！
--- [Epoch 8] ---
--- Train Statistics ---
Loss: 0.0036 (Ent: 0.0078, Head: 0.0016, Tail: 0.0015)
Train F1: 0.7245 | Train precision: 0.8121 | Train recall: 0.6540
--- Val Statistics ---
Loss: 0.0079 (Ent: 0.0169, Head: 0.0037, Tail: 0.0031)
Val F1: 0.5876 | Val precision: 0.5846 | Val recall: 0.5907 | Best Threshold: -1.5
验证集 F1 提升至 0.5876，最佳阈值已更新为 -1.5！
--- [Epoch 9] ---
--- Train Statistics ---
Loss: 0.0028 (Ent: 0.0055, Head: 0.0015, Tail: 0.0014)
Train F1: 0.7463 | Train precision: 0.8871 | Train recall: 0.6441
--- Val Statistics ---
Loss: 0.0079 (Ent: 0.0163, Head: 0.0039, Tail: 0.0033)
Val F1: 0.5941 | Val precision: 0.6276 | Val recall: 0.5640 | Best Threshold: -1.5
验证集 F1 提升至 0.5941，最佳阈值已更新为 -1.5！
--- [Epoch 10] ---
--- Train Statistics ---
Loss: 0.0024 (Ent: 0.0049, Head: 0.0013, Tail: 0.0012)
Train F1: 0.7959 | Train precision: 0.8529 | Train recall: 0.7459
Loss: 0.0082 (Ent: 0.0173, Head: 0.0040, Tail: 0.0034)
Val F1: 0.5944 | Val precision: 0.6084 | Val recall: 0.5812 | Best Threshold: -1.0
验证集 F1 提升至 0.5944，最佳阈值已更新为 -1.0！
--- [Epoch 11] ---
--- Train Statistics ---
Loss: 0.0022 (Ent: 0.0044, Head: 0.0011, Tail: 0.0011)
Train F1: 0.7964 | Train precision: 0.9011 | Train recall: 0.7134
--- Val Statistics ---
Loss: 0.0093 (Ent: 0.0194, Head: 0.0046, Tail: 0.0040)
Val F1: 0.5842 | Val precision: 0.6448 | Val recall: 0.5340 | Best Threshold: -1.0
```
