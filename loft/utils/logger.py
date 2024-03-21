import os
import shutil

from . import get_framework
if get_framework() == "torch":
    from torch import save as SAVE
    from torch import load as LOAD
    from torch import Tensor as TENSOR
    from torch.utils.tensorboard import SummaryWriter as Writer
else:
    import paddle
    from paddle import save as SAVE
    from paddle import load as LOAD
    from paddle import Tensor as TENSOR
    from torch.utils.tensorboard import SummaryWriter as Writer

from . import BaseMethod
from ..data import DataWriter

def _load(model: object = None, address: str = None):
    assert model is not None and address is not None
    if get_framework() == "torch":
        _model_pred = LOAD(address, map_location='cpu')
        model.load_state_dict(_model_pred)
    else:
        _model_pred = LOAD(address, return_numpy=True)
        for _key, _value in _model_pred.items():
            _model_pred[_key] = paddle.to_tensor(_model_pred[_key], place=paddle.CPUPlace()) 
        model.set_state_dict(_model_pred)
    del _model_pred
    return

class Logger(BaseMethod):
    @staticmethod
    def register(config: object = None):
        assert config is not None
        config.register_value("log_addr", ".log")
        config.register_value("checkpoint_addr", ".checkpoint")
        _load_addr = {
                    "model": None,
                    "optimizer": {
                            "key0": None,
                            "key1": None,
                            "key2": None,
                        },
                    "step": None,
                }
        config.register_value("task_name", "tmp")
        config.register_value("load_addr", _load_addr)
        config.register_value("validate_frequency", 1000)
        config.register_value("save_frequency", 10000)
        config.register_value("eval_frequency", 1000)
        config.register_value("eval_writer", DataWriter)
        config.register_value("metrics_frequency", 1)
        config.register_value("dist_frequency", 1)

    def __init__(self, config: object = None):
        self.rank = None
        self.world_size = None
        self.task_name = config.task_name
        self.checkpoint_addr = config.checkpoint_addr
        self.log_addr = config.log_addr
        self.load_addr = config.load_addr
        self.validate_frequency = config.validate_frequency
        self.save_frequency = config.save_frequency
        self.eval_frequency = config.eval_frequency        
        self.eval_writer = DataWriter(config.eval_writer)
        self.metrics_frequency = config.metrics_frequency
        self.dist_frequency = config.dist_frequency

        self.batch_size = config.default.batch_size
        self.validate_batch_size = config.default.validate_batch_size

        self._step = 0
        self.global_rank = 0
        self.world_size = 1

    def new(self, ranks: list = None):
        if ranks is None: 
            if self.global_rank == 0: pass
            else: return
        else:
            if self.global_rank in ranks: pass
            else: return

        _path = os.getcwd()

        self._model_path = os.path.join(_path, self.checkpoint_addr, self.task_name)
        self._log_path = os.path.join(_path, self.log_addr, "{}_{}".format(self.task_name, self.global_rank))

        if self.global_rank == 0:
            if not os.path.exists(self._model_path): os.makedirs(self._model_path)
            else:
                shutil.rmtree(self._model_path)
                os.makedirs(self._model_path)

        if not os.path.exists(self._log_path): os.makedirs(self._log_path)
        else:
            shutil.rmtree(self._log_path)

        self.writer = Writer(self._log_path)
        return

    def set_global_rank(self, rank: int):
        self.global_rank = rank
    def set_world_size(self, rank: int):
        self.world_size = rank

    def step(self, ):
        self._step += 1

    def save(self, model: object = None, optimizer: object = None, ignore_save_frequency: bool = False):
        if self.global_rank != 0: return
        if self._step % self.save_frequency != 0 and not ignore_save_frequency: return

        if model is not None:
            _model_dict = {_k: _v.cpu() for _k, _v in model.state_dict().items()}
            SAVE(_model_dict, os.path.join(self._model_path, "model_{}".format(self._step)))
        if optimizer is not None:
            if type(optimizer) == dict:
                for _key, _value in optimizer.items():
                    _optimizer_dict = {_k: _v.cpu() for _k, _v in _value.state_dict().items()}
                    SAVE(_optimizer_dict, os.path.join(self._model_path, "optimizer_{}_{}".format(_key, self._step)))
            else:
                _optimizer_dict = {_k: _v.cpu() for _k, _v in optimizer.state_dict().items()}
                SAVE(_optimizer_dict, os.path.join(self._model_path, "optimizer_{}".format(self._step)))
        SAVE({"step": self._step}, os.path.join(self._model_path, "step_{}".format(self._step)))

    def load(self, model: object = None, optimizer: object = None, lr_scheduler: object = None):
        assert self.load_addr is not None
        if model is not None:
            _load(model, self.load_addr.model)

        if optimizer is not None:
            if type(optimizer) != dict:
                _load(optimizer, self.load_addr.optimizer)
            else:
                _optimizer_d = self.load_addr.optimizer.to_dict()
                for _key, _value in _optimizer_d:
                    _load(optimizer[_key], _value)

        if lr_scheduler is not None:
            _d = LOAD(self.load_addr.step)
            lr_scheduler._step = _d["step"]
            self._step = _d["step"]

        return

    def metrics_record(self, config: dict = None, rank_ignore: bool = False, *args, **kwargs):
        assert config is not None
        if self.global_rank != 0 and not rank_ignore: return
        if self._step % self.metrics_frequency != 0: return
        for _key, _value in config.items():
            if isinstance(_value, TENSOR):
                self.writer.add_scalar(_key, _value.item(), self._step, *args, **kwargs)
            elif isinstance(_value, (int, float)):
                self.writer.add_scalar(_key, _value, self._step, *args, **kwargs)
            else:
                raise ValueError(f"{_key} with value as {_value} should have type as tensor, int or float!")
        return

    def dist_record(self, config: dict = None, rank_ignore: bool = False, *args, **kwargs):
        assert config is not None
        if self.global_rank != 0 and not rank_ignore: return
        if self._step % self.dist_frequency != 0: return
        for _key, _value in config.items():
             self.writer.add_histogram(_key, _value, self._step, *args, **kwargs)
        return
        
    def eval_record(self, config: dict = None, ):
        assert config is not None
        if self.eval_writer.file is None:
            _file_name = "{}".format(self.global_rank + self.world_size * self._step//self.eval_frequency)
            self.eval_writer.new(_file_name)
        
        _keys = config.keys()
        for _values in zip(*config.values()):
            _d = {}
            for _key, _value in zip(_keys, _values):
                _d[_key] = _value
            self.eval_writer.add(_d)

        if self._step % self.eval_frequency == 0:
            self.eval_writer.close()
            _file_name = "{}".format(self.global_rank + self.world_size * self._step//self.eval_frequency)
            self.eval_writer.new(_file_name)
        return
