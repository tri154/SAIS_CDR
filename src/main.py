import argparse
import torch
from util import logging, set_seed
from trainer import Trainer
from model import Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2004)

    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    
    return args


def train(args, idx_epoch, input_trains, model, optimizer, scheduler):
    model.train()
    optimizer.zero_grad()

    num_batch = math.ceil(len(intputs_train`) / args.batch_size)

    for idx_batch, batch_inputs in enumerate(prepare_batch_train(input_trains, args.batch_size)):
        pass

        
def test():
    pass

def main():
    args = parse_args()
    set_seed(args.seed)
    model = Model()

    best_f1, best_epoch = 0, 0
    for idx_epoch in range(args.num_epoch):

        train()
        epoch_f1 = test()

        if epoch_f1 >= best_f1:
            best_f1, best_epoch = epoch_f1, idx_epoch
            torch.save(model.state_dict(), args.save_path), map_location=

    model.load_state_dict(torch.load(args.save_path, map_location=args.device))
    test()


    
if __name__ == '__main__':
    main()
