import time
import torch
import numpy as np
import random


def logging(text, file):
    file = open(file, 'a')
    print(time.strftime("%Y %b %d %a, %H:%M%S: "),
          time.localtime() + text, file=file, flush=True)
    file.close()


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
