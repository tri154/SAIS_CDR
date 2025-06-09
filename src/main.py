from util import set_seed
from config import Config
from trainer import Trainer
from model import Model
from tester import Tester
    
if __name__ == '__main__':
    cfg = Config
    set_seed(cfg.seed)
    model = Model()
    tester = Tester()
    trainer = Trainer()

    #TODO:
    # do training stuffs here.
