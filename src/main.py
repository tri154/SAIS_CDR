from config import Config
from trainer import Trainer
from model import Model
from tester import Tester
from preprocessing import Preprocessing
    
if __name__ == '__main__':
    cfg = Config()

    pre = Preprocessing(cfg)
    train_set, dev_set, test_set = pre.train_set, pre.dev_set, pre.test_set
    model = Model(cfg).to(cfg.device)
    tester = Tester(cfg, dev_set=dev_set, test_set=test_set)
    trainer = Trainer(cfg, model, train_set=train_set, tester=tester)
    trainer.train()
    # trainer.train(cfg.num_epoches, cfg.batch_size)
