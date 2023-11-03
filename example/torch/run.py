import torch
import numpy as np

from loft import Lofter
from loft.utils import BaseRunner
from loft.utils import Logger
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
            "use_amp": True,
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
        self.use_amp = config.runner.use_amp

    def set_preprocesses(self, preprocesses: dict = None, key: str = None):
        self.datareader.set_preprocesses(preprocesses, key)        

    def set_validate_preprocesses(self, preprocesses: dict = None, key: str = None):
        self.datareader.set_validate_preprocesses(preprocesses, key)

    def train(self, ):
        self.dist_setup()
        if self.use_cudnn: torch.backends.cudnn.benchmark = True
        local_rank, global_rank, world_size = self.get_rank()
        self.datareader.set_global_rank(global_rank)
        self.datareader.set_world_size(world_size)
        self.logger.set_global_rank(global_rank)
        self.logger.set_world_size(world_size)
        self.logger.new(ranks = [0, 1])
        with torch.cuda.device(local_rank):
            self.model.cuda()
            self.model.set_world_size(world_size)
            self.model.set_global_rank(global_rank)
            self.model = self.model_distribute(self.model)
            scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            optimizer = self.optimizer(self.model.parameters(), "AdamW")
            for data in self.datareader.dataloader:
                for _params in optimizer.param_groups:
                    _params["lr"] = self.lr_scheduler()
                
                self.to_cuda(data)
                data["txt"]["input_ids"] = data["txt"]["input_ids"].squeeze()
                data["txt"]["type_ids"] = data["txt"]["type_ids"].squeeze()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _d_items, _d_dist = self.model(data)
                scaler.scale(_d_items["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                logit_scale_clip(self.model)
                _validate_items, _validate_dist = self.validate()
                torch.cuda.synchronize()
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

        with torch.no_grad():
            if self.global_rank == 0:
                self.datareader.validate_dataset.set_data_name("data_0")
                scores = []
                labels = []
                keys = []
                for data in self.datareader.validate_dataloader:
                    self.to_cuda(data)
                    _s = self.model.module.i2i(data)
                    scores += _s
                    labels += data["label"].cpu().detach().numpy().tolist()
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
                    _s = self.model.module.i2t(data)
                    scores += _s
                    labels += data["label"].cpu().detach().numpy().tolist()
                    keys += data["key"]
                _d_items["i2t_auc"] = f_auc(scores, labels)
                _d_items["i2t_pnr"] = f_pnr(scores, labels, keys)
                _d_dist["i2t_score"] = np.array(scores)
            else: pass

        self.datareader.validate_dataset.disable_writer()
        self.model.train()
        torch.distributed.barrier()
        return _d_items, _d_dist            

    def eval(self, ):
        self.dist_setup()
        if self.use_cudnn: cudnn.benchmark = True
        local_rank, global_rank, world_size = self.get_rank()

        self.datareader.set_global_rank(global_rank)
        self.datareader.set_world_size(world_size)
        self.logger.set_global_rank(global_rank)
        self.logger.set_world_size(world_size)

        with torch.cuda.device(local_rank):
            self.model.cuda()
            self.model.set_world_size(world_size)
            self.model = self.model_distribute(self.model)
            self.model.eval()
            for data in self.datareader.dataloader:
                self.to_cuda(data)
                data["txt"]["input_ids"] = data["txt"]["input_ids"].squeeze()
                data["txt"]["type_ids"] = data["txt"]["type_ids"].squeeze()
                _s = self.model.module.i2t(data) 
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
