import os
import io
import time

import gzip
import tarfile

from . import return_transform
from . import BaseMethod

class DataWriter(BaseMethod):
    @staticmethod
    def register(config: object = None):
        _columns = {
            "key": 0,
            "image": 1,
            "txt": 2,
        }
        _transforms = {
            "key": "original",
            "image": "base64_to_bytes",
            "txt": "original",
        } 
        config.register_value("activate", False)
        config.register_value("address", "../yinglong/data_merge4base/wukong_transform")
        config.register_value("format", "tar")
        config.register_value("seq_char", '\t')
        config.register_value("columns", _columns)
        config.register_value("transforms", _transforms)
        config.register_value("idx_mark", False)
        config.register_value("use_original_name", True)

    def __init__(self, config: object):
        self.activate = config.activate
        self.address = config.address
        self.format = config.format
        self.seq_char = config.seq_char
        self.columns = config.columns.to_dict(level = 1)
        self.transforms = config.transforms.to_dict(level = 1)
        self.idx_mark = config.idx_mark
        self.use_original_name = config.use_original_name

        self._transforms = {}
        for _key, _value in self.transforms.items():
            self._transforms[_key] = return_transform(_value)
        self.file = None
        self.file_done = None
        self._idx = None
        self.file_idx = None
        self.filter = self._data_original

    def _data_original(self, data: dict):
        return False

    def new(self, filename: str):
        del self.file
        del self.file_done
        del self._idx
        self._idx = 0
        if self.use_original_name:
            self.filename = os.path.basename(filename).rsplit('.', 1)[0]
        else:
            self.filename = str(self.file_idx)
        if not os.path.exists(self.address): os.mkdirs(self.address)

        if self.format == "txt":
            self.file = open(os.path.join(self.address, "{}.txt".format(self.filename)), 'a')
            self.file_done = "{}.done".format(os.path.join(self.address, "{}.txt".format(self.filename)))
        elif self.format == "gz":
            self.file = gzip.open(os.path.join(self.address, "{}.gz".format(self.filename)), 'at')
            self.file_done = "{}.done".format(os.path.join(self.address, "{}.gz".format(self.filename)))
        elif self.format == "tar":
            self.file = tarfile.open(os.path.join(self.address, "{}.tar".format(self.filename)), 'w:')
            self.file_done = "{}.done".format(os.path.join(self.address, "{}.tar".format(self.filename)))
        elif self.format == "tar.gz":
            self.file = tarfile.open(os.path.join(self.address, "{}.tar.gz".format(self.filename)), 'w:gz')
            self.file_done = "{}.done".format(os.path.join(self.address, "{}.tar.gz".format(self.filename)))
        else:
            self.file = open(os.path.join(self.address, "{}.txt".format(self.filename)), 'a')
            self.file_done = "{}.done".format(os.path.join(self.address, "{}.txt".format(self.filename)))

    def set_file_index(self, rank: int):
        self.file_idx = rank

    def add(self, item: dict):
        if not hasattr(self, "filename"):
            raise ValueError(f"{self.__class__.__name__}'s func as new() should be called before add() operates!")
        if self.filter(item): return
        try:
            if self.format == "txt" or self.format == "gz":

                if self.idx_mark:
                    _mark = self.filename + '_' + self._idx
                    string = "{}{}".format(self.seq_char, _mark)
                else:
                    string = ""

                for _key, _value in item.items():
                    if self._transforms[_key] == "DROP": continue
                    string += "{}{}".format(self.seq_char, self._transforms[_key](_value))
                self.file.write(string[1:]+'\n')
            
            elif self.format == "tar" or self.format == "tar.gz":
                _title = None
                
                if self.idx_mark: _title = "{}".format(self.filename + '_' + str(self._idx))
                else: _title = item["key"]
                
                for _key, _value in item.items():
                    if _key == "key": continue
                    _info = tarfile.TarInfo("{}.{}".format(_title, self.columns[_key]))
                    if self._transforms[_key] == "DROP": continue
                    _value_transformed = self._transforms[_key](_value)
                    _info.size = len(_value_transformed)
                    _info.mtime = time.time()
                    _info.mode = 0o0444
                    _info.uname = ""
                    _info.gname = ""
                    self.file.addfile(_info, io.BytesIO(_value_transformed)) 
        except:
            pass
        self._idx += 1
    
    def close(self, ):
        if self.activate:
            if self.file is None:
                raise ValueError(f"{self.__class__.__name__}'s attribute as file is None! Please set file before {self.__class__.__name__}.close() operates!")
            else:
                self.file.close()
                open(self.file_done, 'a').close()
        else:
            pass
