import os

import gzip
import tarfile

import random
from . import return_transform

class Shuffle(object):
    @staticmethod
    def register(config: object = None):
        config.register_value("frequency", 3)
        config.register_value("quota", 0)
        
    def __init__(self, config: object = None):
        self.quota = config.quota
        self.frequency = config.frequency

        if self.quota is None or self.quota <= 0:
            self.pop = self.original
        else:
            self.idx = 0
            self.idx_ignore_start_pop = 0
            self.list = []
            self.pop = self.random_pick
    
    def original(self, item: dict):
        return item
        
    def random_pick(self, item: dict):
        self.list.append(item)
        if self.idx < self.quota:
            self.idx += 1
            self.idx_ignore_start_pop += 1
            if self.idx > 0 and self.idx_ignore_start_pop % self.frequency == 0:
                _idx = random.randint(0, self.idx-1)
                self.list[_idx], self.list[-1] = self.list[-1], self.list[_idx]
                self.idx -= 1
                return self.list.pop()
            else:
                pass
            return None
        else:
            _idx = random.randint(0, self.quota)
            self.list[_idx], self.list[-1] = self.list[-1], self.list[_idx]
            return self.list.pop()
        

def txt_loop(filename: str, shuffle: object = None, config: object = None):
    def is_valid(_d):
        for _key in _columns:
            if _key == "seq_char": continue
            elif _key not in _d or _columns[_key] is None: return False
            else: pass
        return True

    if not os.path.isfile(filename): return
    assert config is not None and hasattr(config, "columns")
    assert hasattr(config, "transforms")
    assert hasattr(config, "preprocesses")
    assert hasattr(config, "filter")
    assert hasattr(config.columns, "key")
    assert shuffle is not None and hasattr(shuffle, "quota")
    assert hasattr(config, "writer")

    if not hasattr(config, "seq_char"):
        config.register_value("seq_char", '\t')
    if not hasattr(config, "max_split"):
        config.register_value("max_split", -1)
    
    _columns = config.columns.to_dict(level = 1)
    _transforms = config.transforms.to_dict(level = 1)
    _preprocesses = config.preprocesses.to_dict(level = 1)
    _filter = config.filter

    for _key, _value in _transforms.items():
        _transforms[_key] = return_transform(_value)

    if config.writer.activate: config.writer.new(filename)

    if config.format == "txt":
        try:
            file = open(filename, 'r')
        except:
            #raise ValueError(f"The input file:{filename} is failed to open!")
            return
    elif config.format == "gz":
        try:
            file = gzip.open(filename, 'r')
        except:
            #raise ValueError(f"The input file:{filename} is failed to open!")
            return
    else:
        ValueError(f"The format of the {filename} is not consistent with input config!")
    
    _d_idx = -1
    for line in file:
        _d = {}
        _list = line.strip().split(config.seq_char, config.max_split)
        for _key, _idx in _columns.items():
            try:
                _d[_key] = _list[_idx]
            except:
                continue
        if is_valid(_d):
            try:
                _d_idx += 1
                if _d_idx % config.worker_num != config.worker_id: continue
                _item = shuffle.pop(_d)
                if _item is None: 
                    del _d, _list
                    continue
                else:
                    if config.writer.activate: config.writer.add(_item)
                    for _key in _item:
                        _item[_key] = _preprocesses[_key](_transforms[_key](_item[_key]))
                    if _filter(_item): continue
                    yield _item
                    del _d, _list
                    
            except (GeneratorExit, RuntimeError) as e:
                config.writer.close()
                file.close()
                return
            except ValueError as e:
                continue
        else:
            try:
                continue
            except (GeneratorExit, RuntimeError) as e:
                config.writer.close()
                file.close()
                return

    del _columns, _transforms
    config.writer.close()
    file.close()

def tar_loop(filename: str, shuffle: object = None, config: object = None):
    def is_valid(_d):
        for _key in _columns:
            if _key not in d or _columns[_key] is None: return False
            else: pass
        return True

    def reverse_dict():
        _d_reversed = {}
        for _key, _value in _columns.items():
            _d_reversed[_value] = _key
        return _d_reversed
    
    assert config is not None and hasattr(config, "columns")
    assert hasattr(config, "transforms")
    assert hasattr(config, "preprocesses")
    assert hasattr(config, "filter")
    assert shuffle is not None and hasattr(shuffle, "quota")
    assert hasattr("config", "writer")

    if not hasattr(config.columns, "key"):
        config.columns.register_value("key", "title")
    
    if config.writer.activate: config.writer.new(filename)
    _columns = config.columns.to_dict(level = 1)
    _reversed_columns = reverse_dict()
    _columns_set = set(_columns.values())
    _transforms = config.transforms.to_dict(level = 1)
    _filter = config.filter

    for _key, _value in _transforms.items():
        _transforms[_key] = return_transform(_value)

    if config.format == "tar":
        try:
            file = tarfile.open(filename, 'r')
        except:
            return
            #raise ValueError(f"The input file:{filename} is failed to open!")
    elif config.format == "tar.gz":
        try:
            file = tarfile.open(filename, 'r:gz')
        except:
            return
            #raise ValueError(f"The input file:{filename} is failed to open!")
    else:
        ValueError(f"The format of the {filename} is not consistent with input config!")

    pres_title = None
    _d_idx = -1
    for item in file:
        _d = {}
        cur_title = item.name.rsplit('.', 1)[0]
        cur_format = item.name.rsplit('.', 1)[-1]
        if cur_format not in _columns_set: continue
        if pres_title != cur_title:
            if pres_title is None:
                _d[_columns["key"]] = cur_title
            else:
                if is_valid(_d):
                    try:
                        _d_idx += 1
                        if _d_idx % config.worker_num != config.worker_id: continue 

                        _item = shuffle.pop(_d)
                        if _item is None:
                            del _d, cur_title, cur_format
                        else:
                            if config.writer.activate:
                                _d_ori_key = {}
                                for _key, _value in _columns.items():
                                    _d_ori_key[_key] = _item[_value]
                                config.writer.add(_d_ori_key)

                            _d_transformed = {}
                            for _key, _value in _columns.items():
                                _d_transformed[_key] = _preprocesses[_key](_transforms[_reversed_columns[_key]](_item[_value]))
                            if _filter(_d_transformed): continue
                            yield _d_transformed
                            del _d_transformed, _d, cur_title, cur_format
                            
                    except (GeneratorExit, RuntimeError) as e:
                        config.writer.close()
                        file.close()
                        return
                    except ValueError as e:
                        continue
                else:
                    try:
                        pass
                    except (GeneratorExit, RuntimeError) as e:
                        config.writer.close()
                        file.close()
                        return
                _d = {}
                _d["key"] = cur_title
            pres_title = cur_title
        try:
            _content = file.extractfile(item).read()
            _d[cur_format] = _content
        except:
            continue
        

    del _columns, _columns_set, _transforms
    config.writer.close()
    file.close()

def return_loop(_format: str):
    if _format == "txt" or _format == "gz":
        return txt_loop
    elif _format == "tar" or _format == "tar.gz":
        return tar_loop
    else:
        return txt_loop
