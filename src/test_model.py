import argparse
import torch
from config import Config
from model import Model
from tester import Tester
from preprocessing import Preprocessing

def setup():
    args = argparse.Namespace()

    args.dataset = "cdr"
    args.save_path = "best.pt"
    args.log_path = "log.txt"
    args.seed = 2004

    args.num_epoch = 30
    args.batch_size = 4
    args.update_freq = 1
    args.warmup_ratio = 0.06
    args.max_grad_norm = 1.0

    # Suggest hyperparameters with optuna
    args.new_lr = 1e-4
    args.pretrained_lr = 1.472039003976042e-05
    args.adam_epsilon = 1e-6

    args.device = "cuda:0"
    args.transformer = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    args.type_dim = 20
    args.graph_layers = 3

    args.use_psd = True
    args.lower_temp = 2.0
    args.upper_temp = 20.0
    args.loss_tradeoff = 4.999979907145212

    args.use_sc = True
    args.sc_temp = 0.16096448806072833
    args.sc_weight = 0.1509642367395748

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pt", type=str)
    args = parser.parse_args()

    cfg = Config(setup())
    
    pre = Preprocessing(cfg)
    train_set, dev_set, test_set = pre.train_set, pre.dev_set, pre.test_set
    model = Model(cfg).to(cfg.device)
    model.load_state_dict(torch.load(args.model_pt, map_location=cfg.device))
    tester = Tester(cfg, dev_set=dev_set, test_set=test_set)
    P, R, F1 = tester.test(model, dataset='dev')
    print(f"Dev set: P={P}, R={R}, F1={F1}")
    P, R, F1 = tester.test(model, dataset='test')
    print(f"Test set: P={P}, R={R}, F1={F1}")
