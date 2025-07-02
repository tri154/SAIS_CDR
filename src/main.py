from config import Config
from trainer import Trainer
from model import Model
from tester import Tester
from preprocessing import Preprocessing
from config import setup, cleanup
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch

def train_DDP(rank, cfg, pre):
    is_ddp = cfg.world_size != 1
    if is_ddp:
        setup(rank, cfg.world_size)
        cfg.device = rank

    train_set, dev_set, test_set = pre.train_set, pre.dev_set, pre.test_set

    if is_ddp:
        # Make sure the divide is perfect (same length across ranks).
        train_set = train_set[int(rank * len(train_set) / cfg.world_size): int((rank + 1) * len(train_set) / cfg.world_size)]
        dev_set = dev_set[int(rank * len(dev_set) / cfg.world_size): int((rank + 1) * len(dev_set) / cfg.world_size)]
        test_set = test_set[int(rank * len(test_set) / cfg.world_size): int((rank + 1) * len(test_set) / cfg.world_size)]

    model = Model(cfg).to(cfg.device)
    if is_ddp:
        model = DDP(model, device_ids=[rank], ignore_non_tensor_inputs=True)

    tester = Tester(cfg, dev_set=dev_set, test_set=test_set)
    trainer = Trainer(cfg, model, train_set=train_set, tester=tester)
    trainer.train(cfg.num_epoch, cfg.batch_size)
    # trainer.debug()

    if is_ddp:
        cleanup()

if __name__ == '__main__':
    cfg = Config()
    pre = Preprocessing(cfg)

    if cfg.world_size == 1:
        train_DDP(rank=0, cfg=cfg)
    else:
        mp.spawn(train_DDP,
                 args=(cfg, pre),
                 nprocs=cfg.world_size,
                 join=True)

    # pre = Preprocessing(cfg)
    # train_set, dev_set, test_set = pre.train_set, pre.dev_set, pre.test_set

    # model = Model(cfg).to(cfg.device)
    # tester = Tester(cfg, dev_set=dev_set, test_set=test_set)
    # trainer = Trainer(cfg, model, train_set=train_set, tester=tester)
    # trainer.train(cfg.num_epoch, cfg.batch_size)
    # trainer.debug()

        
