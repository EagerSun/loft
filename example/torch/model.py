import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as checkpoint
from collections import OrderedDict

from loft import BaseModel

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        _ori_type = x.dtype
        _res = super().forward(x.type(torch.float32))
        return _res.type(_ori_type)

class GELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Block(BaseModel):
    @staticmethod
    def register(config: object = None):
        config.register_value("width", 768)
        config.register_value("head", 12)
        config.register_value("dropout_rate", 0.0)
    def __init__(self, config: object = None):
        super().__init__()
        self.width = config.width
        self.head = config.head
        self.dropout_rate = config.dropout_rate
        self.attn = nn.MultiheadAttention(self.width, self.head, dropout=self.dropout_rate)
        self.mlp = nn.Sequential(
                    OrderedDict([
                        ("fc", nn.Linear(self.width, self.width * 4)),
                        ("gelu", GELU()),
                        ("proj", nn.Linear(self.width * 4, self.width)),
                        ])
                )
        self.ln_pre = LayerNorm(self.width)
        self.ln_post = LayerNorm(self.width)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, attn_mask: torch.Tensor = None):
        x = self.ln_pre(x)
        x = x + self.attn(x, x, x, need_weights=False, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        x = self.ln_post(x)
        x = self.mlp(x)
        return x

class Transformer(BaseModel):
    @staticmethod
    def register(config: object = None, ):
        config.register_value("layer", 12)
        config.register_value("recompute", True)
        config.register_value("block", Block)

    def __init__(self, config: object = None, ):
        super().__init__()
        self.layer = config.layer
        self.recompute = config.recompute
        self.blocks = nn.Sequential(*[
                Block(config.block)
                for _ in range(self.layer)
            ])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, attn_mask: torch.Tensor = None):
        if self.recompute:
            for _idx in range(self.layer):
                x = checkpoint(self.blocks[_idx], x, key_padding_mask, attn_mask)     
        else:
            for _idx in range(self.layer):
                x = self.blocks[_idx](x, key_padding_mask, attn_mask)
        return x

class VisualTransformer(BaseModel):
    @staticmethod
    def register(config: object = None, ):
        config.register_value("input_size", 224)
        config.register_value("patch_size", 14)
        config.register_value("input_channel", 3)
        config.register_value("embed_dim", 256)
        config.register_value("transformer", Transformer)

    def __init__(self, config: object = None):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.transformer = Transformer(config.transformer)

        _width = config.transformer.block.width
        _scale = _width ** -0.5
        self.conv = nn.Conv2d(in_channels=config.input_channel, out_channels=_width, kernel_size=config.patch_size, stride=config.patch_size, bias=False)

        self.class_embedding = nn.Parameter(_scale * torch.randn(_width))
        self.positional_embedding = nn.Parameter(_scale * torch.randn((config.input_size // config.patch_size) ** 2+1, _width))
        self.ln_pre = LayerNorm(_width)
        self.ln_post = LayerNorm(_width)

        self.proj = nn.Parameter(_scale * torch.randn(_width, self.embed_dim))
        self.proj_final = nn.Parameter(_scale * torch.randn(self.embed_dim, self.embed_dim))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, attn_mask: torch.Tensor = None, ):
        x = self.conv(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros((x.shape[0], 1, x.shape[-1]), dtype = x.dtype, device = x.device), x], dim = 1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj
        x = self.relu(x)
        x = x @ self.proj_final
        return x 

class TextTransformer(BaseModel):
    @staticmethod
    def register(config: object = None,):
        config.register_value("vocab_size", 50000)
        config.register_value("context_length", 30)
        config.register_value("transformer", Transformer)
        config.register_value("sentence_field", 4)
        config.register_value("embed_dim", 256)
        config.register_value("dropout_rate", 0.01)

    def __init__(self, config: object = None,):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.transformer.block.width)
        self.sent_embedding = nn.Embedding(config.sentence_field, config.transformer.block.width)
        self.position_embedding = nn.Parameter(torch.randn((config.context_length, config.transformer.block.width)))
        self.proj = nn.Linear(config.transformer.block.width, config.transformer.block.width)
        self.proj_final = nn.Linear(config.transformer.block.width, config.embed_dim)
        self.ln_pre = LayerNorm(config.transformer.block.width)
        self.dropout_pre = nn.Dropout(p=config.dropout_rate)
        self.transformer = Transformer(config.transformer)
        self.tanh = nn.Tanh()

    def build_input_mask(self, src_ids):
        return torch.where(src_ids == 0, 1, 0).type(torch.bool)

    def forward(self, x, type_ids):
        _x = self.token_embedding(x)
        key_padding_mask = self.build_input_mask(x)
        x = _x + self.position_embedding
        x = x + self.sent_embedding(type_ids)
        x = self.ln_pre(x)
        x = self.dropout_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, key_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.proj(x[:, 0, :])
        x = self.tanh(x)
        x = self.proj_final(x)
        return x

class Model(BaseModel):
    @staticmethod
    def register(config: object = None):
        config.register_value("visual", VisualTransformer)
        config.register_value("text", TextTransformer)
        config.register_value("logit_scale", 0.07)

    def __init__(self, config: object = None):
        super().__init__()
        self.visual = VisualTransformer(config.visual)
        self.text = TextTransformer(config.text)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1./config.logit_scale))

    def set_world_size(self, rank: int = None):
        self.world_size = rank
    def set_global_rank(self, rank: int = None):
        self.global_rank = rank

    @torch.no_grad()
    def logit_scale_clip(self, ):
        self.logit_scale = torch.clamp(self.logit_scale, 0, 4.6052)

    def forward(self, data):
        image = data["image"]
        text, type_ids = data["txt"]["input_ids"], data["txt"]["type_ids"]
        image_features = self.visual(image)
        text_features = self.text(text, type_ids)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        loss, logits, labels = self._loss(image_features, text_features)
        accuracy_image = _accuracy(logits, labels)
        accuracy_text = _accuracy(logits.T, labels)
        _d_item = {
                "loss": loss,
                "accuracy_image": accuracy_image,
                "accuracy_text": accuracy_text,
                "logit_scale": self.logit_scale,
                }
        logits_diag = logits.diagonal()
        _d_dist = {
                "logits_diag": logits_diag.cpu().detach().numpy()
                }
        return _d_item, _d_dist

    def _loss(self, image_features: torch.Tensor, text_features: torch.Tensor):
        _image_features = torch.empty(self.world_size, image_features.shape[0], image_features.shape[1], dtype=image_features.dtype, device=image_features.device)
        _text_features = torch.empty(self.world_size, text_features.shape[0], text_features.shape[1], dtype=text_features.dtype, device=text_features.device)

        l_image = list(_image_features.unbind(0))
        l_text = list(_text_features.unbind(0))

        torch.distributed.all_gather(l_image, image_features)
        torch.distributed.all_gather(l_text, text_features)

        l_image[self.global_rank] = image_features
        l_text[self.global_rank] = text_features

        image_features_all = torch.cat(l_image)
        text_features_all = torch.cat(l_text)
        logits = self.logit_scale.exp() * image_features_all @ text_features_all.T
        labels = torch.arange(image_features_all.shape[0], device=image_features.device)
        image_loss = torch.nn.functional.cross_entropy(logits, labels)
        text_loss = torch.nn.functional.cross_entropy(logits.T, labels)
        loss = (image_loss + text_loss) / 2
        return loss, logits, labels 

    @torch.no_grad()
    def i2i(self, data):
        image_query = data["image_q"]
        image_recall = data["image_r"]
        feature_query = self.visual(image_query)
        feature_recall = self.visual(image_recall)
        feature_query = feature_query / feature_query.norm(dim=-1, keepdim=True)
        feature_recall = feature_recall / feature_recall.norm(dim=-1, keepdim=True)
        logits_diag = (feature_query @ feature_recall.T).diagonal()
        return logits_diag.cpu().detach().numpy().tolist()

    @torch.no_grad()
    def i2t(self, data):
        image_query = data["image_q"]
        text_recall = data["txt"]["input_ids"]
        type_ids_recall = data["txt"]["type_ids"]
        feature_image = self.visual(image_query)
        feature_text = self.text(text_recall, type_ids_recall)
        feature_image = feature_image / feature_image.norm(dim=-1, keepdim=True)
        feature_text = feature_text / feature_text.norm(dim=-1, keepdim=True)
        logits_diag = (feature_image @ feature_text.T).diagonal()
        return logits_diag.cpu().detach().numpy().tolist()

@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor):
    _batch = logits.shape[0]
    _, _idx = logits.topk(1, 1)
    _idx = _idx.T
    correct = _idx.eq(labels.view(1, -1)).view(-1).float().sum(0, ) * 100./_batch
    return correct

@torch.no_grad()
def logit_scale_clip(model: nn.Module = None, ):
    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)
