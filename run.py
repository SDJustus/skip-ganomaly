"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

import time
from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model
import torch
import numpy as np
from lib.models.skipganomaly import seed



##
def main():
    """ Training
    """
    opt = Options().parse()
    opt.print_freq = opt.batchsize
    seed(opt.manualseed)
    print("Seed:", str(torch.seed()))
    if opt.phase == "inference":
        opt.batchsize=1
    data = load_data(opt)
    model = load_model(opt, data)
    if opt.phase == "inference":
        model.inference()
    else:
        train_start = time.time()
        model.train()
        train_time = time.time() - train_start
        print (f'Train time: {train_time} secs')

if __name__ == '__main__':
    main()
