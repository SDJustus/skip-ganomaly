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
    data = load_data(opt)
    model = load_model(opt, data)
    if opt.phase == "inference":
        model.inference()
    else:
        model.train()
    

if __name__ == '__main__':
    main()
