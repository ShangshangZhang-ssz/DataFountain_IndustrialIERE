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