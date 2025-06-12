from config import Config
from trainer import Trainer
from model import Model
from tester import Tester
from preprocessing import Preprocessing
    
if __name__ == '__main__':
    cfg = Config()

    pre = Preprocessing(cfg)
    train_set, dev_set, test_set = pre.train_set, pre.dev_set, pre.test_set
    model = Model(cfg)#TODO
    tester = None #TODO
    trainer = Trainer(cfg, model, train_set=train_set, tester=tester)
    trainer.debug()

    #TODO:
    # do training stuffs here.
