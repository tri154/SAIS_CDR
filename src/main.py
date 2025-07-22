from config import Config
from trainer import Trainer
from model import Model
from tester import Tester
from preprocessing import Preprocessing

def run_training(cfg, no_tqdm=False):
    pre = Preprocessing(cfg)
    train_set, dev_set, test_set = pre.train_set, pre.dev_set, pre.test_set
    model = Model(cfg).to(cfg.device)
    tester = Tester(cfg, dev_set=dev_set, test_set=test_set)
    trainer = Trainer(cfg, model, train_set=train_set, tester=tester)
    best_dev_f1 = trainer.train(cfg.num_epoch, cfg.train_batch_size, no_tqdm=no_tqdm)
    return best_dev_f1
    # trainer.debug()


if __name__ == '__main__':
    cfg = Config()
    run_training(cfg, no_tqdm=True)
