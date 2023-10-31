import os
import math

import re
import six
import collections
from functools import partial

from lofter import BaseMethod, get_framework

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

if get_framework() == "torch":
    import torch
    from torchvision import transforms
else:
    import paddle
    from paddle.vision import transforms

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

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file, 'rb')
    for num, line in enumerate(fin):
        items = convert_to_unicode(line.strip()).split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab

class WordTokenizer(object):
    def __init__(self, vocab_file, context_length, do_lower_case=True, sp_vocab=False):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pat = re.compile(r'([a-zA-Z0-9]+\s|\S+\s|[a-zA-Z0-9]+$|\S+$)')
        self.do_lower_case = do_lower_case
        self.sp_vocab = sp_vocab
        self.pad = '[PAD]'
        self.unk = '[UNK]'
        self.cls = '[CLS]'
        self.pad_id = self.vocab.get(self.pad)
        self.cls_id = self.vocab.get(self.cls)
        self.max_length = context_length

    def wordpiece(self, token, vocab, unk_token, sp_vocab=False):
        """call with single word"""
        chars = list(token.strip())
        max_input_chars_per_word = 1024
        if len(chars) > max_input_chars_per_word:
            return [unk_token], [(0, len(chars))]

        is_bad = False
        start = 0
        sub_tokens = []
        sub_pos = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start == 0 and sp_vocab:
                    substr = u'\u2581' + substr
                if start > 0 and not sp_vocab:
                    if re.match("^[A-Za-z0-9]+$", substr):
                        substr = "##" + substr
                    else:
                        substr = substr
                if substr in vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            sub_pos.append((start, end))
            start = end
        if is_bad:
            return [unk_token], [(0, len(chars))]
        else:
            return sub_tokens, sub_pos

    def word_token(self, text):
        if len(text) == 0:
            return []
        text = convert_to_unicode(text)
        if self.do_lower_case:
            text = text.lower()
        res = []
        for match in self.pat.finditer(text):
            words, _ = self.wordpiece(
                match.group(0),
                vocab=self.vocab,
                unk_token=self.unk,
                sp_vocab=self.sp_vocab)
            res.extend(words)
        #print(res)
        return res

    def convert_tokens_to_ids(self, tokens):
        #print(tokens)
        return [self.vocab.get(t, self.vocab[self.unk]) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab.get(i) for i in ids]
    
    def padding_to_max(self, one_list):
        max_length_m = self.max_length - 1
        return [self.cls_id] + one_list[:max_length_m] + [self.pad_id] * max(0, max_length_m - len(one_list))

    def decode(self, tokens):
        if paddle.is_tensor(tokens):
            tokens = paddle.tolist(tokens)
        tokens = [token for token in tokens if token not in (0,)]
        return ''.join(self.convert_ids_to_tokens(tokens))
    
    def encode(self, text):
        return paddle.to_tensor(self.convert_tokens_to_ids(self.word_token(text)))

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        tensor_list = []
        type_ids = []
        for one_line in texts:
            ids = self.convert_tokens_to_ids(self.word_token(one_line))
            padding_res = self.padding_to_max(ids)
            tensor_list.append(padding_res)
            type_ids.append([int(i != self.pad_id and i != self.cls_id) for i in padding_res])
        #input_ids = paddle.to_tensor(tensor_list)
        return tensor_list, type_ids


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
