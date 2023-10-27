import os

from . import get_framework
if get_framework() == "torch":
    from torch.utils.data import IterableDataset, get_worker_info
else:
    from paddle.io import IterableDataset, get_worker_info

from . import BaseMethod
from . import return_loop, Shuffle, Address, ValidateAddress

class ValidateDataset(IterableDataset):
    @staticmethod
    def register(config: object = None, ):
        config.register_value("address", ValidateAddress)
        config.register_value("shuffle", Shuffle)

    def __init__(self, config: object = None):
        self.address = ValidateAddress(config.address)
        self.shuffle = Shuffle(config.shuffle)

        self.data_name = None

    def set_preprocesses(self, preprocesses: dict = None, key: str = None):
        assert preprocesses is not None
        self.address.set_preprocesses(preprocesses = preprocesses, key = key)

    def set_filter(self, filter: object = None, key: str = None):
        assert filter is not None
        self.address.set_filter(filter = filter, key = key)
    def set_writer_filter(self, filter: object = None, key: str = None):
        assert filter is not None
        self.address.set_writer_filter(filter = filter, key = key)

    def set_global_rank(self, rank: int):
        self.address.set_global_rank(rank)
        #return self
    def set_world_size(self, rank: int):
        self.address.set_world_size(rank)
        #return self

    def set_data_name(self, name: str):
        self.data_name = name
        #return self

    def disable_writer(self, key: str = None):
        self.address.disable_writer(key = key)

    def __iter__(self, ):
        if self.data_name is None:
            raise ValueError(f"The {self.__class__.__name__}'s attribute: data_name should be set by func: set_data_name")
            
        _worker_info = get_worker_info()
        if _worker_info is not None:
            self.address.set_worker_id(_worker_info.id)
            self.address.set_worker_num(_worker_info.num_workers)

        self.address.new()
        for _file, _config in self.address.pop(self.data_name):
            for _item in return_loop(_config.format)(_file, self.shuffle, _config):
                yield _item

        return

class Dataset(IterableDataset):
    @staticmethod
    def register(config: object = None, ):
        config.register_value("address", Address)
        config.register_value("shuffle", Shuffle)

    def __init__(self, config: object = None):
        self.address = Address(config.address)
        self.shuffle = Shuffle(config.shuffle)

    def set_preprocesses(self, preprocesses: dict = None, key: str = None):
        assert preprocesses is not None
        self.address.set_preprocesses(preprocesses = preprocesses, key = key)

    def set_filter(self, filter: object = None, key: str = None):
        assert filter is not None
        self.address.set_filter(filter = filter, key = key)
    def set_writer_filter(self, filter: object = None, key: str = None):
        assert filter is not None
        self.address.set_writer_filter(filter = filter, key = key)

    def set_global_rank(self, rank: int):
        self.address.set_global_rank(rank)
        #return self
    def set_world_size(self, rank: int):
        self.address.set_world_size(rank)
        #return self

    def __iter__(self, ):
        _worker_info = get_worker_info()
        if _worker_info is not None:
            self.address.set_worker_id(_worker_info.id)
            self.address.set_worker_num(_worker_info.num_workers)
        for _file, _config in self.address.pop():
            for _item in return_loop(_config.format)(_file, self.shuffle, _config):
                yield _item

        return
