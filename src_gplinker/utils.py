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