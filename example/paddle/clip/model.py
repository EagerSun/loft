import math
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute as checkpoint
from collections import OrderedDict

from loft import BaseModel

class LayerNorm(nn.LayerNorm):
    def forward(self, x: paddle.Tensor):
        _ori_type = x.dtype
        _res = super().forward(x.astype(paddle.float32))
        return _res.astype(_ori_type)

class GELU(BaseModel):
    def forward(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(1.702 * x)

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
        self.attn = nn.MultiHeadAttention(self.width, self.head, dropout=self.dropout_rate)
        self.mlp = nn.Sequential(
                        ("fc", nn.Linear(self.width, self.width * 4)),
                        ("gelu", GELU()),
                        ("proj", nn.Linear(self.width * 4, self.width)),
                )
        self.ln_pre = LayerNorm(self.width)
        self.ln_post = LayerNorm(self.width)

    def forward(self, x: paddle.Tensor, attn_mask: paddle.Tensor = None):
        x = self.ln_pre(x)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape((x.shape[0], 1, 1, x.shape[1])).expand([-1, self.head, -1, -1])
        x = x + self.attn(x, x, x, attn_mask=attn_mask, )
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
        self.blocks = nn.LayerList([
                Block(config.block)
                for _ in range(self.layer)
            ])

    def forward(self, x: paddle.Tensor, attn_mask: paddle.Tensor = None):
        if self.recompute:
            for _idx in range(self.layer):
                x = checkpoint(self.blocks[_idx], x, attn_mask)     
        else:
            for _idx in range(self.layer):
                x = self.blocks[_idx](x, attn_mask)
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
        self.conv = nn.Conv2D(in_channels=config.input_channel, out_channels=_width, kernel_size=config.patch_size, stride=config.patch_size, bias_attr=False)

        self.class_embedding = self.create_parameter(shape=[_width], 
                default_initializer=paddle.nn.initializer.Assign(_scale * paddle.randn(paddle.to_tensor(_width))))

        _init_positional_embedding = _scale * paddle.randn(paddle.to_tensor(_width))
        self.positional_embedding = self.create_parameter(shape=_init_positional_embedding.shape,
                default_initializer=paddle.nn.initializer.Assign(_init_positional_embedding))
        self.ln_pre = LayerNorm(_width)
        self.ln_post = LayerNorm(_width)

        self.proj = self.create_parameter(shape=[_width, self.embed_dim],
                default_initializer=paddle.nn.initializer.Assign(_scale * paddle.randn((_width, self.embed_dim))))
        self.proj_final = self.create_parameter(shape=[self.embed_dim, self.embed_dim],
                default_initializer=paddle.nn.initializer.Assign(_scale * paddle.randn((self.embed_dim, self.embed_dim))))
        self.relu = nn.ReLU()

    def forward(self, x: paddle.Tensor, attn_mask: paddle.Tensor = None, ):
        x = self.conv(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.transpose((0, 2, 1))
        x = paddle.concat([self.class_embedding.astype(x.dtype) + paddle.zeros((x.shape[0], 1, x.shape[-1]), dtype = x.dtype), x], axis = 1)
        x = x + self.positional_embedding.astype(x.dtype)
        x = self.ln_pre(x)
        x = x.transpose((1, 0, 2))
        x = self.transformer(x)
        x = x.transpose((1, 0, 2))
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
        self.token_embedding = nn.Embedding(config.vocab_size, config.transformer.block.width, 
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02)))
        self.sent_embedding = nn.Embedding(config.sentence_field, config.transformer.block.width,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02)))
        self.position_embedding = self.create_parameter(shape=[config.context_length, config.transformer.block.width],
                default_initializer=nn.initializer.TruncatedNormal(std=0.02))
        self.proj = nn.Linear(config.transformer.block.width, config.transformer.block.width,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02)))
        self.proj_final = nn.Linear(config.transformer.block.width, config.embed_dim,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02)))
        self.ln_pre = LayerNorm(config.transformer.block.width,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.)),
                bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.)))
        self.dropout_pre = nn.Dropout(p=config.dropout_rate)
        self.transformer = Transformer(config.transformer)
        self.tanh = nn.Tanh()

    @property
    def dtype(self):
        return self.position_embedding.dtype

    def build_input_mask(self, src_ids):
        _mask = paddle.equal(src_ids, 0).astype(paddle.bool)
        _mask = (~_mask).astype(self.dtype)
        inf_t = paddle.ones_like(_mask) * float('-inf')
        zero_t = paddle.zeros_like(_mask)
        res = paddle.where(_mask<1,x=inf_t, y=_mask)
        res = paddle.where(res==1, zero_t, res)
        return res

    def forward(self, x, type_ids):
        _x = self.token_embedding(x)
        attn_mask = self.build_input_mask(x)
        x = _x + self.position_embedding
        x = x + self.sent_embedding(type_ids)
        x = self.ln_pre(x)
        x = self.dropout_pre(x)
        x = x.transpose((1, 0, 2))
        x = self.transformer(x, attn_mask)
        x = x.transpose((1, 0, 2))
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
        self.logit_scale = self.create_parameter(shape=[1],
            default_initializer=paddle.nn.initializer.Constant(math.log(1. / config.logit_scale))) 

    def set_world_size(self, rank: int = None):
        self.world_size = rank
    def set_global_rank(self, rank: int = None):
        self.global_rank = rank

    @paddle.no_grad()
    def logit_scale_clip(self, ):
        self.logit_scale = paddle.clip(self.logit_scale, 0, 4.6052)

    def forward(self, data, choice=None):
        if choice is None: pass
        elif choice == "i2i":
            return self.i2i(data)
        elif choice == "i2t":
            return self.i2t(data)
        else: pass

        image = data["image"]
        text, type_ids = data["txt"]["input_ids"], data["txt"]["type_ids"]
        image_features = self.visual(image)
        text_features = self.text(text, type_ids)
        image_features = image_features / image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / text_features.norm(axis=-1, keepdim=True)
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

    def _loss(self, image_features: paddle.Tensor, text_features: paddle.Tensor):
        l_image = []
        l_text = []

        paddle.distributed.all_gather(l_image, image_features)
        paddle.distributed.all_gather(l_text, text_features)

        l_image[self.global_rank] = image_features
        l_text[self.global_rank] = text_features

        image_features_all = paddle.concat(l_image)
        text_features_all = paddle.concat(l_text)
        logits = self.logit_scale.exp() * image_features_all @ text_features_all.T
        labels = paddle.arange(image_features_all.shape[0], dtype=paddle.int64)
        image_loss = paddle.nn.functional.cross_entropy(logits, labels)
        text_loss = paddle.nn.functional.cross_entropy(logits.T, labels)
        loss = (image_loss + text_loss) / 2
        return loss, logits, labels 

    @paddle.no_grad()
    def i2i(self, data):
        image_query = data["image_q"]
        image_recall = data["image_r"]
        feature_query = self.visual(image_query)
        feature_recall = self.visual(image_recall)
        feature_query = feature_query / feature_query.norm(axis=-1, keepdim=True)
        feature_recall = feature_recall / feature_recall.norm(axis=-1, keepdim=True)
        logits_diag = (feature_query @ feature_recall.T).diagonal()
        return logits_diag.cpu().detach().squeeze().tolist()

    @paddle.no_grad()
    def i2t(self, data):
        image_query = data["image_q"]
        text_recall = data["txt"]["input_ids"]
        type_ids_recall = data["txt"]["type_ids"]
        feature_image = self.visual(image_query)
        feature_text = self.text(text_recall, type_ids_recall)
        feature_image = feature_image / feature_image.norm(axis=-1, keepdim=True)
        feature_text = feature_text / feature_text.norm(axis=-1, keepdim=True)
        logits_diag = (feature_image @ feature_text.T).diagonal()
        return logits_diag.cpu().detach().squeeze().tolist()

@paddle.no_grad()
def _accuracy(logits: paddle.Tensor, labels: paddle.Tensor):
    _batch = logits.shape[0]
    _, _idx = logits.topk(1, 1)
    _idx = _idx.T
    correct = paddle.equal(_idx, labels.reshape([1, -1])).reshape([-1]).astype(paddle.float32).sum(0, keepdim=True)
    correct *= 100./_batch
    return correct

@paddle.no_grad()
def logit_scale_clip(model: nn.Layer = None, ):
    if isinstance(model, paddle.DataParallel):
        _buffer = model._layers.logit_scale.clip(0, 4.6052)
        _buffer._share_buffer_to(model._layers.logit_scale)
    else:
        model.logit_scale.clip(0, 4.6052)
        _buffer._share_buffer_to(model.logit_scale)
