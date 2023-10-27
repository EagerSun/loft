import os

from . import get_framework
if get_framework() == "torch":
    from torch.nn import Module as Module
else:
    from paddle.nn import Layer as Module

import json
import yaml
import shutil
import types

class Unit(object):
    def __init__(self, ):
        return

class Lofter(object):
    def __init__(self, non_default: bool = False):
        if not non_default:
            self.default = Unit()
        return

    def register_value(self, name: str, value, default_as: str = None):
        if isinstance(value, dict):
            _tmp = Lofter()
            _tmp.default = self.default
            for _key, _value in value.items():
                _tmp.register_value(_key, _value)
            setattr(self, name, _tmp)
        elif isinstance(value, (type(None), int, float, str, list, tuple, types.MethodType, types.FunctionType)):
            setattr(self, name, value)
            if default_as is not None:
                setattr(self.default, default_as, value)
        else:
            _tmp = Lofter()
            _tmp.default = self.default
            setattr(self, name, _tmp)
            try:
                value.register(_tmp)
            except:
                raise NotImplementedError(f"The register func of {value.__class__.__name__} are not implemented!")
        return

    def to_dict(self, save_default: bool = False, save_func: bool = True, level: int = None, index: int = 0):
        _d = dict()
        
        if level is None: pass
        else:
            if level == index: 
                return self
            else:
                pass

        for _key, _value in self.__dict__.items():
            if _key == "default":
                if save_default: _d[_key] = _value.__dict__
            elif isinstance(_value, (type(None), int, float, str, list, tuple, types.MethodType, types.FunctionType)):
                if isinstance(_value, (types.MethodType, types.FunctionType)):
                    if save_func: _d[_key] = _value
                else:
                     _d[_key] = _value
            else:
                try:
                    _d[_key] = _value.to_dict(save_default = False, save_func = save_func, level = level, index = index + 1)
                except:
                    _type = type(_d[_key])
                    raise ValueError(f"The type as {_type} are not implemented with to_dict() func, the target type should be Lofter!")
        return _d

    def from_dict(self, url = None, *args, **kwargs):
        _d = None
        #if url is None: raise ValueError(f"The input url:{url} should be dict or address of yaml/json, but None!")
        if isinstance(url, dict):
            _d = url
        elif url is None:
            _d = self.load()
        else:
            pass

        if _d is None:
            raise ValueError(f"Load conf dictionary failed!")

        if "default" in _d:
            self.default = Lofter(non_default = True)
            for _k, _v in _d["default"].items():
                self.default.register_value(_k, _v)

        for _key, _value in _d.items():
            if _key == "default":
                continue
            else:
                self.register_value(_key, _value)
        return

    def merge(self, url = None, *args, **kwargs):
        def _merge(_d, _d_target):
            for _key in _d_target:
                if _key in _d:
                    if isinstance(_d_target[_key], (type(None), int, float, str, list, tuple, types.MethodType, types.FunctionType)):
                        if isinstance(_d[_key], (type(None), int, float, str, list, tuple, types.MethodType, types.FunctionType)):
                            _d_target[_key] = _d[_key]
                    elif isinstance(_d_target[_key], dict):
                        if isinstance(_d[_key], dict):
                            _merge(_d[_key], _d_target[_key])
                    else:
                        raise ValueError(f"The types of dict's value should be included in (int, float, str, list, tuple, dict)")
            return
            
        _d = None
        
        if isinstance(url, dict):
            _d = url
        elif url is None:
            _d = self.load()
        else:
            pass
        
        if _d is None: raise ValueError(f"Load conf dictionary input url:{url} is failed!")
        _d_target = self.to_dict(save_default = True, save_func = True)
        _merge(_d, _d_target)
        self.from_dict(_d_target)
        return

    def save(self, url: str = None, split: bool = True, *args, **kwargs):
        if os.path.exists("./.conf"): 
            shutil.rmtree("./.conf")
        os.makedirs("./.conf")

        _d = self.to_dict(save_default = True, save_func = False)
        
        if url is not None and (url.endswith(".json") or url.endswith(".yaml")):
            if url.endswith(".json"):
                string = json.dumps(_d, *args, **kwargs)
                with open(os.path.join("./.conf", url), 'w') as file:
                    file.write(string)
            elif url.endswith(".yaml") or url.endswith(".yml"):
                with open(os.path.join("./.conf", url), 'w') as file:
                    yaml.dump(_d, file, sort_keys=False, *args, **kwargs)
        else:
            if split:
                for _key in _d:
                    with open(os.path.join("./.conf", _key+".yaml"), 'w') as file:
                        yaml.dump(_d[_key], file, sort_keys=False, *args, **kwargs)
            else:
                with open(os.path.join("./.conf", "conf.yaml"), 'w') as file:
                    yaml.dump(_d, file, sort_keys=False, *args, **kwargs)
        del _d
        return

    def load(self, *args, **kwargs):
        def return_dict(url: str, *args, **kwargs):
            if url.endswith(".json"):
                with open(url, 'r') as file:
                    return json.load(file)
            elif url.endswith(".yaml") or url.endswith(".yml"):
                with open(url, 'r') as file:
                    return yaml.unsafe_load(file, *args, **kwargs) 
        
        if not os.path.exists("./.conf"):
            raise RuntimeError(f"The default conf address of this project as .conf is not exist!")
            return
        if len(os.listdir("./.conf")) == 0:
            raise RuntimeError(f"The default conf address of this project as .conf have no valid file!")
            return

        _d = {}   
        
        if len(os.listdir("./.conf")) == 1:
            _item = os.listdir("./.conf")[0]
            _d = return_dict(os.path.join("./.conf", _item))
        else:
            for _item in os.listdir("./.conf"):
                _key = _item.split('.')[0]
                _d[_key] = return_dict(os.path.join("./.conf", _item), *args, **kwargs)
        if not _d:
            raise ValueError(f"The result conf dict loaded from .conf is not valid!")
        return _d
        
class BaseMethod(object):
    @staticmethod
    def register(config: object = None):
        raise NotImplementedError(f"The register func should be implemented!")

    def __init__(self, config: object = None):
        return

    def new_func(self, name, func):
        """
        example:
        def func(self, *args):
            self.*args are included!!!
            operations are edited here!!!
            return

        usage
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

class BaseModel(Module):
    @staticmethod
    def register(config: object, ):
        raise NotImplementedError(f"The register function should be implemented!")
    def __init__(self, ):
        super().__init__()
