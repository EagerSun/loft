import os
import math

import re
import six
import collections
from functools import partial

from loft import BaseMethod, get_framework

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

if get_framework() == "torch":
    import torch
    from torchvision import transforms
else:
    import paddle
    from paddle.vision import transforms

# Optimizer defined here

class Optimizer(BaseMethod):
    @staticmethod
    def register(config: object = None):
        assert config is not None
        config.register_value("betas1", 0.9)
        config.register_value("betas2", 0.98)
        config.register_value("eps", 1.0e-8)
        config.register_value("weight_decay", 0.02)

    def __init__(self, config: object = None):
        self.adamW = partial(torch.optim.AdamW, betas=(config.betas1, config.betas2), eps=config.eps, weight_decay=config.weight_decay)
        return

    def __call__(self, model_params: object = None, key: str = None):
        _d_optimizer = {"AdamW": self.adamW}
        if key is None:
            for _key in _d_optimizer:
                _d_optimizer[_key] = _d_optimizer[_key](model_params)
            return _d_optimizer
        else:
            _d_optimizer[key] = _d_optimizer[key](model_params)
            return _d_optimizer[key]

# Lr_scheduler defined here

class LR_Scheduler(BaseMethod):
    @staticmethod
    def register(config: object = None):
        assert config is not None
        config.register_value("total_steps", 100000)
        config.register_value("warmup_steps", 2000)
        config.register_value("start_lr", 0.0)
        config.register_value("max_lr", 0.00001)
        config.register_value("final_lr", 0.0)

        config.register_value("exp_T1", 0.000095)

        config.register_value("_lr_scheduler", "linear")
    
    def __init__(self, config: object):
        self.total_steps = config.total_steps
        self.warmup_steps = config.warmup_steps
        self.start_lr = config.start_lr
        self.max_lr = config.max_lr
        self.final_lr = config.final_lr

        self.exp_T1 = config.exp_T1

        self._lr_scheduler = config._lr_scheduler

        self._lr = None
        self._step = 0
        
        self._scheduler = None
        if self._lr_scheduler == "linear":
            self._scheduler = self.linear_scheduler
        elif self._lr_scheduler == "cosine":
            self._scheduler = self.cosine_scheduler
        elif self._lr_scheduler == "exp":
            self._scheduler = self.exp_scheduler
        else:
            self._scheduler = self.linear_scheduler

    def step(self, ):
        self._lr = None
        self._step += 1
    def is_stop(self, ):
        return self.total_steps <= self._step
    def stop(self, ):
        return self.is_stop()

    def linear_scheduler(self, ):
        if self._lr is None:
            if self._step+1 <= self.warmup_steps:
                _lr = (self.max_lr - self.start_lr) / self.warmup_steps * (self._step+1) + self.start_lr
            else:
                _lr = self.max_lr + (self.final_lr - self.max_lr) / (self.total_steps - self.warmup_steps) * (self._step+1 - self.warmup_steps)

            if _lr <= 0.0:
                _lr = 0.0
            self._lr = _lr
            return _lr
        else:
            return self._lr
    
    def cosine_scheduler(self, ):
        if self._lr is None:
            if self._step+1 <= self.warmup_steps:
                _lr = (self.max_lr - self.start_lr) / self.warmup_steps * (self._step+1) + self.start_lr
            else:
                _lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * \
                    (1 + math.cos(math.pi * (self._step+1 - self.warmup_steps) / (self.total_steps - self.warmup_steps)))

            if _lr <= 0.0:
                _lr = 0.0
            self._lr = _lr
            return _lr
        else:
            return self._lr 

    def exp_scheduler(self, ):
        if self._lr is None:
            if self._step+1 <= self.warmup_steps:
                _lr = (self.max_lr - self.start_lr) / self.warmup_steps * (self._step+1) + self.start_lr
            else:
                _B = (self.max_lr - self.final_lr)/(1 - math.exp(-1 * self.exp_T1 * (self.total_steps - self.warmup_steps-1)))
                _b = (self.final_lr - self.max_lr * math.exp(-1 * self.exp_T1 * (self.total_steps - self.warmup_steps-1))) \
                        / (1 - math.exp(-1 * self.exp_T1 * (self.total_steps - self.warmup_steps-1)))
                _lr = _B * math.exp(-1 * self.exp_T1 * (self._step - self.warmup_steps)) + _b

            if _lr <= 0.0:
                _lr = 0.0
            self._lr = _lr
            return _lr
        else:
            return self._lr

    def __call__(self, ):
        return self._scheduler()

# WordToken

class WordTokenizer(object):
    def __init__(self, ):
        pass

# Validate metrics

def f_auc(scores: list, labels: list):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def f_pnr(scores: list, labels: list, keys: list):
    def _pnr(scores: list, labels: list):
        length = len(scores)
        _p, _n = 0, 0
        for i in range(0, length):
            for j in range(i+1, length):
                if labels[i] == labels[j]: continue
                elif (labels[i] - labels[j]) * (scores[i] - scores[j]) > 0: _p += 1
                elif (labels[i] - labels[j]) * (scores[i] - scores[j]) < 0: _n += 1
        return _p, _n

    _d = {}
    pos_num = neg_num = 0
    for _key, _score, _label in zip(keys, scores, labels):
        if _key in _d:
            _d[_key]["score"].append(_score)
            _d[_key]["label"].append(_label)
        else:
             _d[_key] = {}
             _d[_key]["score"] = [_score]
             _d[_key]["label"] = [_label]
    for _key in _d:
        _p, _n = _pnr(_d[_key]["score"], _d[_key]["label"])
        pos_num += _p
        neg_num += _n
    if not neg_num: return 10
    else: return float(pos_num/neg_num)    

# Preprocess defined for training/validating

def _preprocesses():
    
    _transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    _tokenizer = WordTokenizer(vocab_file="./.vocab.txt", context_length=30)
    
    def key(string: str = None):
        assert string is not None
        return string
    def image(string: object = None):
        assert string is not None
        return _transform(string)
    def txt(string: str = None):
        assert string is not None
        ids, type_ids = _tokenizer(string)
        return {"input_ids": torch.as_tensor(ids), "type_ids": torch.as_tensor(type_ids)}
    return {"key": key, "image": image, "txt": txt,}

def _validate_preprocesses():
    _transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    _tokenizer = WordTokenizer(vocab_file="./.vocab.txt", context_length=30)

    def key(string: str = None):
        assert string is not None
        return string
    def image(string: object = None):
        assert string is not None
        return _transform(string)
    def txt(string: str = None):
        assert string is not None
        ids, type_ids = _tokenizer(string)
        return {"input_ids": torch.as_tensor(ids), "type_ids": torch.as_tensor(type_ids)}
    def label(num: int = None):
        assert num is not None
        if num > 1: return 1
        else: return 0
    i2i = {"key": key, "image_q": image, "image_r": image, "key_r": key, "label": label}
    i2t = {"key": key, "image_q": image, "txt": txt, "label": label} 
    return {"data_0": i2i, "data_1": i2t}
