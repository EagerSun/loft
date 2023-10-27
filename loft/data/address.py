import os
import re

import random
from datetime import date

from . import return_transform
from . import BaseMethod, DataWriter

class DataFolder(BaseMethod):
    @staticmethod
    def register(config: object = None):
        _address = "../yinglong/data_merge4base/wukong/part-{00000..09999}"
        _columns = {
            "key": 0,
            "image": 1,
            "txt": 2,
        }
        _transforms = {
            "key": "original",
            "image": "base64_to_Image",
            "txt": "original",
        }
        _seq_char = '\t'
        _max_split = -1
        config.register_value("address", _address)
        config.register_value("columns", _columns)
        config.register_value("transforms", _transforms)
        config.register_value("format", "txt")
        config.register_value("seq_char", '\t')
        config.register_value("max_split", -1)
        config.register_value("writer", DataWriter) 
        
    def __init__(self, config: object = None):
        self.address = config.address
        self.columns = config.columns
        self.transforms = config.transforms
        self.format = config.format
        self.seq_char = config.seq_char
        self.max_split = config.max_split
        self.writer = DataWriter(config.writer)
        self.worker_num = 1
        self.worker_id = 0
        self.filter = self._data_original
        return
    
    def _data_original(self, data: dict):
        return False

    def set_worker_num(self, rank: int = 1):
        self.worker_num = rank
    def set_worker_id(self, rank: int = 0):
        self.worker_id = rank

class ValidateAddress(BaseMethod):
    @staticmethod
    def register(config: object = None):
        for i in range(3):
            config.register_value("data_{}".format(i), DataFolder)
        return

    def __init__(self, config: object = None, ):
        assert config is not None
        _d = config.to_dict(level = 1)

        self._d = {}
        for _key, _value in _d.items():
            self._d[_key] = DataFolder(_value)

        self.filelist = {}
        self.global_rank = 0
        self.world_size = 1

        self.worker_id = 0
        self.worker_num = 1

    def set_preprocesses(self, preprocesses: dict = None, key: str = None):
        assert preprocesses is not None
        if key is not None:
            self._d[key].add_attr("preprocesses", preprocesses)
        else:
            for _key in self._d:
                self._d[_key].add_attr("preprocesses", preprocesses)

    def set_filter(self, filter: object = None, key: str = None):
        assert filter is not None
        if key is not None:
            self._d[key].add_attr("filter", filter)
        else:
            for _key in self._d:
                self._d[_key].add_attr("filter", filter)

    def set_writer_filter(self, filter: object = None, key: str = None):
        assert filter is not None
        if key is not None:
            self._d[key].writer.add_attr("filter", filter)
        else:
            for _key in self._d:
                self._d[_key].writer.add_attr("filter", filter)

    def set_global_rank(self, rank: int = None):
        self.global_rank = rank
    def set_world_size(self, rank: int = None):
        self.world_size = rank

    def set_worker_id(self, rank: int = 0):
        self.worker_id = rank
    def set_worker_num(self, rank: int = 1):
        self.worker_num = rank

    def disable_writer(self, key: str = None):
        if key is not None:
            self._d[key].writer.activate = False
        else:
            for _key in self._d:
                self._d[_key].writer.activate = False

    def new(self, ):
        del self.filelist
        self.filelist = {}
        pattern = re.compile(r"\d+\.\.\d+")
        for _key, _value in self._d.items():
            _format_name = os.path.basename(_value.address)
            _format_folder = os.path.dirname(_value.address)
            _value.add_attr("folder_address", _format_folder)
            if pattern.search(_format_name) is None:
                self.filelist[_key] = ["{}/{}".format(_key, _format_name)]
            else:
                _start, _end = pattern.search(_format_name).group(0).split("..")
                _format_name_pre, _format_name_post = re.split(r"{\d+\.\.\d+}", _format_name)

                if str(int(_start)) == _start: _length = 0
                else: _length = len(_start)
                for i in range(int(_start), int(_end)+1):
                    _idx = str(i).zfill(_length)
                    _file_name = "{}/{}".format(_key, _format_name_pre+_idx+_format_name_post)
                    if _key not in self.filelist: self.filelist[_key] = [_file_name]
                    else: self.filelist[_key].append(_file_name)
    
    def pop(self, dataname: str, ):
        if not self.filelist:
            self.new()
        else:
            pass

        assert dataname in self.filelist
        if len(self.filelist[dataname]) == 1:
            for idx, item in enumerate(self.filelist[dataname]):
                _address = self._d[os.path.dirname(item)].folder_address
                if self.worker_num > 1: self._d[os.path.dirname(item)].writer.activate = False

                self._d[os.path.dirname(item)].set_worker_num(self.worker_num)
                self._d[os.path.dirname(item)].set_worker_id(self.worker_id)
                self._d[os.path.dirname(item)].writer.set_file_index(idx)
                yield os.path.join(_address, os.path.basename(item)), self._d[os.path.dirname(item)]
            
        else:
            for idx, item in enumerate(self.filelist[dataname][(self.global_rank**self.worker_num+self.worker_id)::(self.world_size*self.worker_num)]):
                _address = self._d[os.path.dirname(item)].folder_address
                _idx = (self.global_rank*self.worker_num+self.worker_id) + idx * (self.world_size*self.worker_num)
                self._d[os.path.dirname(item)].writer.set_file_index(_idx)
                yield os.path.join(_address, os.path.basename(item)), self._d[os.path.dirname(item)]
            
        return 
        
class Address(BaseMethod):
    @staticmethod
    def register(config: object = None):
        for i in range(3):
            config.register_value("data_{}".format(i), DataFolder)
        return

    def __init__(self, config: object = None, ):
        assert config is not None
        _d = config.to_dict(level = 1)

        self._d = {}
        for _key, _value in _d.items():
            self._d[_key] = DataFolder(_value)

        self.seed = None
        self.filelist = []

        self.global_rank = 0
        self.world_size = 1

        self.worker_id = 0
        self.worker_num = 1

    def set_preprocesses(self, preprocesses: dict = None, key: str = None):
        assert preprocesses is not None
        if key is not None:
            self._d[key].add_attr("preprocesses", preprocesses)
        else:
            for _key in self._d:
                self._d[_key].add_attr("preprocesses", preprocesses) 

    def set_filter(self, filter: object = None, key: str = None):
        assert filter is not None
        if key is not None:
            self._d[key].add_attr("filter", filter)
        else:
            for _key in self._d:
                self._d[_key].add_attr("filter", filter)

    def set_writer_filter(self, filter: object = None, key: str = None):
        assert filter is not None
        if key is not None:
            self._d[key].writer.add_attr("filter", filter)
        else:
            for _key in self._d:
                self._d[_key].writer.add_attr("filter", filter) 

    def set_global_rank(self, rank: int = None):
        self.global_rank = rank
    def set_world_size(self, rank: int = None):
        self.world_size = rank

    def set_worker_id(self, rank: int = 0):
        self.worker_id = rank
    def set_worker_num(self, rank: int = 1):
        self.worker_num = rank

    def _seeds(self):
        if self.seed is None: 
            _today_Ymd = int(date.today().strftime("%Y%m%d"))
            _today_md = int(date.today().strftime("%m%d"))
            self.seed = _today_Ymd % _today_md
        rng = random.Random(self.seed)
        self.seed = rng.randint(0, 10000)

    def new(self, ):
        
        if not self.filelist:
            pattern = re.compile(r"\d+\.\.\d+")
            for _key, _value in self._d.items():
                _format_name = os.path.basename(_value.address)
                _format_folder = os.path.dirname(_value.address)
                _value.add_attr("folder_address", _format_folder)
                _start, _end = pattern.search(_format_name).group(0).split("..")
                _format_name_pre, _format_name_post = re.split(r"{\d+\.\.\d+}", _format_name)
                
                if str(int(_start)) == _start: _length = 0
                else: _length = len(_start)
                for i in range(int(_start), int(_end)+1):
                    _idx = str(i).zfill(_length)
                    _file_name = "{}/{}".format(_key, _format_name_pre+_idx+_format_name_post)
                    self.filelist.append(_file_name)

        self._seeds()
        random.Random(self.seed).shuffle(self.filelist)

    def pop(self):
        while True:
            self.new()
            for idx, item in enumerate(self.filelist[(self.global_rank*self.worker_num+self.worker_id)::(self.world_size*self.worker_num)]):
                _address = self._d[os.path.dirname(item)].folder_address
                _idx = (self.global_rank*self.worker_num+self.worker_id) + idx * (self.world_size*self.worker_num)
                self._d[os.path.dirname(item)].writer.set_file_index(_idx)
                yield os.path.join(_address, os.path.basename(item)), self._d[os.path.dirname(item)]
            for _value in self._d.values(): _value.writer.activate = False
        return
                
