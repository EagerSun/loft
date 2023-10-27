import os

from . import get_framework
if get_framework() == "torch":
    from torch.utils.data import DataLoader
else:
    from paddle.io import DataLoader

from . import BaseMethod, ValidateDataset, Dataset

class DataReader(BaseMethod):
    @staticmethod
    def register(config: object = None):
        config.register_value("batch_size", 256, default_as = "batch_size")
        config.register_value("num_workers", 1)
        config.register_value("dataset", Dataset)
        
        config.register_value("validate_batch_size", 256, default_as = "validate_batch_size")
        config.register_value("validate_num_workers", 1)
        config.register_value("validate_dataset", ValidateDataset)

    def __init__(self, config: object = None, ):
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self._step = 0

        self.dataset = Dataset(config.dataset)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size = self.batch_size, 
                                     num_workers = self.num_workers)
        
        self.validate_batch_size = config.validate_batch_size
        self.validate_num_workers = config.validate_num_workers
        
        self.validate_dataset = ValidateDataset(config.validate_dataset)
        self.validate_dataloader = DataLoader(self.validate_dataset,
                                    batch_size = self.validate_batch_size,
                                    num_workers = self.validate_num_workers)

    def set_preprocesses(self, preprocesses: object = None, key: str = None):
        assert preprocesses is not None
        if self.dataset is not None:
            self.dataset.set_preprocesses(preprocesses = preprocesses, key = key)
        return

    def set_validate_preprocesses(self, preprocesses: object = None, key: str = None):
        assert preprocesses is not None
        if self.validate_dataset is not None:
            self.validate_dataset.set_preprocesses(preprocesses = preprocesses, key = key)
        return

    def set_filter(self, filter: object = None, key: str = None):
        """
        key format should be formed as:
            e.g. data_0, validate.data_0...
        """
        assert filter is not None
        if key is None or not key.startswith("validate"):
            if self.dataset is not None:
                self.dataset.set_filter(filter = filter, key = key)
        else:
            if self.validate_dataset is not None:
                _key = key.strip().split('.')[-1]
                if not _key or _key == "validate": _key = None
                self.validate_dataset.set_filter(filter = filter, key = _key)
        return

    def set_writer_filter(self, filter: object = None, key: str = None):
        """
        key format should be formed as:
            e.g. data_0, validate.data_0...
        """
        assert filter is not None
        if key is None or not key.startswith("validate"):
            if self.dataset is not None:
                self.dataset.set_writer_filter(filter = filter, key = key)
        else:
            if self.validate_dataset is not None:
                _key = key.strip().split('.')[-1]
                if not _key or _key == "validate": _key = None
                self.validate_dataset.set_writer_filter(filter = filter, key = _key)
        return

    def set_global_rank(self, rank: int):
        self.dataset.set_global_rank(rank)
        self.validate_dataset.set_global_rank(rank)
        return self
    def set_world_size(self, rank: int):
        self.dataset.set_world_size(rank)
        self.validate_dataset.set_world_size(rank)
        return self 

    def step(self, ):
        self._step += 1

