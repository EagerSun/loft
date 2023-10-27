import os

from . import get_framework, get_numgpu_per_node
if get_framework() == "torch":
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.distributed import init_process_group as init_parallel
    from torch import Tensor as TENSOR
else:
    import paddle.distributed as dist
    from paddle import DataParallel as DistributedDataParallel
    from paddle.distributed import init_parallel_env as init_parallel
    from paddle import Tensor as TENSOR

from . import BaseMethod
#from datareader import DataReader
#from logger import Logger
#from example import Model
#from utils import Optimizer, LR_Scheduler

class BaseRunner(BaseMethod):
    @staticmethod
    def register(config: object = None):
        #config.register_value("datareader", DataReader)
        #config.register_value("logger", Logger)
        #config.register_value("model", Model)
        #config.register_value("optimizer", Optimizer)
        #config.register_value("lr_scheduler", LR_Scheduler)
        print(config.to_dict())
        pass
        
    def __init__(self, config: object = None):
        #self.datareader = DataReader(config.dataReader)
        #self.logger = Logger(config.logger)
        #self.model = Model(config.model)
        #self.optimizer = Optimizer(config.optimizer)
        #self.lr_scheduler = LR_Scheduler(config.lr_scheduler)
        self.distribute_url = None
        self.set_framework()
        self.gpu_per_node = None
        self.global_rank = None
        self.local_rank = None
        self.world_size = None
        return
    
    def set_framework(self, ):
        self.framework = get_framework()
        return

    def dist_setup(self, ):
        if self.framework == "torch":
            if self.distribute_url is None:
                raise ValueError(f"{self.__class__.__name__}'s attribute self.distribute_url should be set with valid value but None!")
            _, global_rank, world_size = self.get_rank()
            init_parallel(backend="nccl", init_method=self.distribute_url, rank=global_rank, world_size=world_size)
        else:
            init_parallel()
        return

    def dist_destory(self, ):
        dist.destroy_process_group()
        return
 
    def model_distribute(self, model: object = None):
        assert model is not None
        return DistributedDataParallel(model)

    def get_rank(self, ):
        if self.gpu_per_node is None:
            self.gpu_per_node = get_numgpu_per_node()
        
        if self.world_size is None or self.global_rank is None:
            if self.framework == "torch":
                self.world_size = int(os.environ['WORLD_SIZE'])
                self.global_rank = int(os.environ['RANK'])
                self.local_rank = int(os.environ["LOCAL_RANK"])
            else:
                self.global_rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.local_rank = self.global_rank % self.gpu_per_node
        else:
            pass
        return self.local_rank, self.global_rank, self.world_size
    
    def to_cuda(self, item: dict = None):
        assert item is not None
        assert self.local_rank is not None
        if self.framework == "torch":
            for _key, _value in item.items():
                if isinstance(_value, TENSOR): 
                    item[_key] = _value.cuda("cuda:{}".format(self.local_rank))
                elif isinstance(_value, dict): self.to_cuda(_value)
                else: pass
        else:
            pass
        return

    def set_preprocesses(self, preprocesses: dict = None):
        assert preprocesses is not None
        raise NotImplementedError(f"{self.__class__.__name__}'s set_preprocesses should be implemented!")
    def set_validate_preprocesses(self, preprocesses: dict = None, key: str = None):
        assert preprocesses is not None
        raise NotImplementedError(f"{self.__class__.__name__}'s set_validate_preprocesses should be implemented!")

    def set_filter(self, filter: object = None, key: str = None):
        return
    def set_writer_filter(self, filter: object = None, key: str = None):
        return

    def train(self, ):
        raise NotImplementedError(f"{self.__class__.__name__}'s train should be implemented!")
        #pass
    def validate(self, ):
        raise NotImplementedError(f"{self.__class__.__name__}'s validate should be implemented!")
        #pass
    def eval(self, ):
        raise NotImplementedError(f"{self.__class__.__name__}'s eval should be implemented!")
        #pass
    
