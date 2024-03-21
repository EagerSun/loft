import paddle
from paddle.distributed import fleet
from paddle.distributed import init_parallel_env as init_parallel
from paddle.distributed.sharding import group_sharded_parallel
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
try:
    from paddle.fluid.dygraph.parallel import sync_params_buffers
except ImportError:
    from paddle.distributed.parallel import sync_params_buffers

import numpy as np

from loft import Lofter, Logger, BaseRunner
from loft.data import DataReader

from model import Model, logit_scale_clip
from utils import Optimizer, LR_Scheduler
from utils import _preprocesses, _validate_preprocesses
from utils import f_auc, f_pnr

class Runner(BaseRunner):
    @staticmethod
    def register(config: object = None):
        _runner = {
            "distribute_url": None,
            "use_cudnn": True,
            "gpu_per_node": 2,
            "amp_type": "O2",
            "sharding_strategy": "HYBRID_ZERO2",
        }
        config.register_value("logger", Logger)
        config.register_value("datareader", DataReader)
        config.register_value("model", Model)
        config.register_value("optimizer", Optimizer)
        config.register_value("lr_scheduler", LR_Scheduler)
        config.register_value("runner", _runner)

    def __init__(self, config: dict = None):
        super().__init__()
        self.logger = Logger(config.logger)
        self.datareader = DataReader(config.datareader)
        self.model = Model(config.model)
        self.optimizer = Optimizer(config.optimizer)
        self.lr_scheduler = LR_Scheduler(config.lr_scheduler)
        self.distribute_url = config.runner.distribute_url
        self.use_cudnn = config.runner.use_cudnn
        self.amp_type = config.runner.amp_type
        self.sharding_strategy = config.runner.sharding_strategy

    def dist_setup(self, mode = "train"):
        if mode == "train":
            strategy = fleet.DistributedStrategy()
            _dp_degree = None
            _sharding_degree = None
            if "HYBRID" in self.sharding_strategy:
                _dp_degree = self.node_num
                _sharding_degree = self.gpus_per_node
            else:
                _dp_degree = 1
                _sharding_degree = self.world_size
            
            strategy.hybrid_configs = {
                "dp_degree": _dp_degree,
                "sharding_degree": _sharding_degree,
            }
            strategy.find_unused_parameters = True
            
            fleet.init(is_collective=True, strategy=strategy)
            
            _hcg = fleet.get_hybrid_communicate_group()
            _dp_rank = _hcg.get_data_parallel_rank()
            _sharding_rank = _hcg.get_sharding_parallel_rank()
            return        
        else:
            init_parallel()

    def set_preprocesses(self, preprocesses: dict = None, key: str = None):
        self.datareader.set_preprocesses(preprocesses, key)        

    def set_validate_preprocesses(self, preprocesses: dict = None, key: str = None):
        self.datareader.set_validate_preprocesses(preprocesses, key)

    def train(self, ):
        self.initial_rank()
        self.dist_setup()
        self.datareader.set_global_rank(self.global_rank)
        self.datareader.set_world_size(self.world_size)
        self.logger.set_global_rank(self.global_rank)
        self.logger.set_world_size(self.world_size)
        self.logger.new(ranks = [0, 1])
        self.model.set_world_size(self.world_size)
        self.model.set_global_rank(self.global_rank)
        scaler = paddle.amp.GradScaler(init_loss_scaling=65536)
        optimizer = self.optimizer(self.model.parameters(), "AdamW")
        
        if self.amp_type != "None":
            self.model = paddle.amp.decorate(
                    models = model,
                    level=self.amp_type)

        _hcg = fleet.get_hybrid_communicate_group()
        _dp_group = _hcg.get_data_parallel_group()
        _sharding_group = _hcg.get_sharding_parallel_group()
        sync_params_buffers(self.model,
                comm_group=_dp_group,
                src_rank=_dp_group.ranks[0])
    
        if "ZERO1" in self.sharding_strategy:
            _level = "os"
        elif "ZERO2" in self.sharding_strategy:
            _level = "os_g"
        else:
            _level = "p_g_os"

        try:
            self.model, optimizer, scaler = group_sharded_parallel(
                    self.model, optimizer, level, 
                    scaler=scaler, group=_sharding_group, 
                    dp_group=_dp_group if _dp_group.nranks > 1 else None)
        except:
            self.model, optimizer, scaler = group_sharded_parallel(
                    self.model, optimizer, level,
                    scaler=scaler, group=_sharding_group)

        for data in self.datareader.dataloader:
            optimizer.set_lr(self.lr_scheduler())
            
            self.to_cuda(data)
            data["txt"]["input_ids"] = data["txt"]["input_ids"].squeeze()
            data["txt"]["type_ids"] = data["txt"]["type_ids"].squeeze()
            with paddle.amp.auto_cast(enable=self.use_amp):
                _d_items, _d_dist = self.model(data)
            _scale = self.scaler.scale(_d_items["loss"])
            _scale.backward()
            fused_allreduce_gradients(list(self.model.parameters()), None)
            #scaler.minimize(optimizer, _scale)
            scaler.step(optimizer)
            optimizer.clear_grad()
            logit_scale_clip(self.model)
            _validate_items, _validate_dist = self.validate()
            paddle.device.cuda.synchronize()
            _d_items["lr"] = self.lr_scheduler()
            self.logger.metrics_record(_d_items)
            self.logger.dist_record(_d_dist)
            if _validate_items:
                self.logger.metrics_record(_validate_items, rank_ignore = True)
            if _validate_dist:
                self.logger.dist_record(_validate_dist, rank_ignore = True)
           
            #self.logger.save(self.model, optimizer, ignore_save_frequency = True)

            self.datareader.step()
            self.lr_scheduler.step()
            self.logger.step()
            if self.lr_scheduler.stop(): 
                self.logger.save(self.model, optimizer, ignore_save_frequency = True)
                break           
        self.dist_destory()

    def validate(self, ):
        _d_items = {}
        _d_dist = {}
        if self.lr_scheduler._step % self.logger.validate_frequency != 0: 
            return _d_items, _d_dist

        self.model.eval()

        with paddle.no_grad():
            if self.global_rank == 0:
                self.datareader.validate_dataset.set_data_name("data_0")
                scores = []
                labels = []
                keys = []
                for data in self.datareader.validate_dataloader:
                    self.to_cuda(data)
                    _s = self.model(data, choice="i2i")
                    scores += _s
                    labels += data["label"].cpu().detach().squeeze().tolist()
                    keys += data["key"]
                _d_items["i2i_auc"] = f_auc(scores, labels)
                _d_items["i2i_pnr"] = f_pnr(scores, labels, keys)
                _d_dist["i2i_score"] = np.array(scores)

            elif self.global_rank == 1:
                self.datareader.validate_dataset.set_data_name("data_1")
                scores = []
                labels = []
                keys = []
                for data in self.datareader.validate_dataloader:
                    self.to_cuda(data)
                    data["txt"]["input_ids"] = data["txt"]["input_ids"].squeeze()
                    data["txt"]["type_ids"] = data["txt"]["type_ids"].squeeze()
                    _s = self.model(data, choice="i2t")
                    scores += _s
                    labels += data["label"].cpu().detach().squeeze().tolist()
                    keys += data["key"]
                _d_items["i2t_auc"] = f_auc(scores, labels)
                _d_items["i2t_pnr"] = f_pnr(scores, labels, keys)
                _d_dist["i2t_score"] = np.array(scores)

            else: pass
    
        self.datareader.validate_dataset.disable_writer()
        self.model.train()
        paddle.distributed.barrier()
        return _d_items, _d_dist            

    def eval(self, ):
        self.initial_rank()
        self.dist_setup()
        if self.use_cudnn: cudnn.benchmark = True

        self.datareader.set_global_rank(self.global_rank)
        self.datareader.set_world_size(self.world_size)
        self.logger.set_global_rank(self.global_rank)
        self.logger.set_world_size(self.world_size)
        self.model.set_global_rank(self.global_rank)
        self.model.set_world_size(self.world_size)
        self.model = self.model_distribute(self.model)
        self.model.eval()
        for data in self.datareader.dataloader:
            self.to_cuda(data)
            data["txt"]["input_ids"] = data["txt"]["input_ids"].squeeze()
            data["txt"]["type_ids"] = data["txt"]["type_ids"].squeeze()
            _s = self.model.i2t(data) 
            _d_data = {
                    "key": data["key"],
                    "score": _s, 
                }
            self.logger.eval_record(_d_data)
        
        self.datareader.step()
        self.logger.step()
        return


if __name__ == "__main__":
    config = Lofter()
    #Runner.register(config)
    #config.save()
    _train_preprocesses = _preprocesses()
    _validate_preprocesses = _validate_preprocesses()
    config.from_dict()
    runner = Runner(config)
    runner.set_preprocesses(_train_preprocesses)
    runner.set_validate_preprocesses(_validate_preprocesses["data_0"], "data_0")
    runner.set_validate_preprocesses(_validate_preprocesses["data_1"], "data_1") 
    runner.train()
