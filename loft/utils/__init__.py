from .. import get_framework, get_numgpu_per_node

if get_framework() == "torch":
    from torch.nn import Module as Module
else:
    from paddle.nn import Layer as Module

from .logger import Logger
from .runner import BaseRunner

class BaseModel(Module):
    @staticmethod
    def register(config: object, ):
        raise NotImplementedError(f"The register function should be implemented!")
    def __init__(self, ):
        super().__init__()

class BaseMethod(object):
    @staticmethod
    def register(config: object = None):
        raise NotImplementedError(f"The register func should be implemented!")

    def __init__(self, config: object = None):
        return

    def add_func(self, name, func):
        """
        example:
        def func(self, *args):
            self.*args are included!!!
            operations are edited here!!!
            return

        usage:
        object.func(object, *args)
        """
        setattr(self, name, func)
        return

    def add_attr(self, name: str, value: object):
        if isinstance(value, dict):
            _tmp = BaseMethod()
            for _key, _value in value.items():
                _tmp.add_attr(_key, _value)
            setattr(self, name, _tmp)
        elif isinstance(value, (type(None), int, float, str, list, tuple, types.MethodType, types.FunctionType)):
            setattr(self, name, value)
        else:
            setattr(self, name, value)
        return

    def to_dict(self, level: int = None, index: int = 0):
        _d = dict()

        if level is None: pass
        else:
            if level == index:
                return self
            else:
                pass

        for _key, _value in self.__dict__.items():
            if isinstance(_value, (type(None), int, float, str, list, tuple, types.MethodType, types.FunctionType)):
                if isinstance(_value, (types.MethodType, types.FunctionType)):
                    _d[_key] = _value
            else:
                try:
                    _d[_key] = _value.to_dict(level = level, index = index + 1)
                except:
                    _type = type(_d[_key])
                    raise ValueError(f"object as type {_type} are not implemented to_dict() func, the target type should be BaseMethod!")
        return _d
