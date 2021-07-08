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

def seed(seed_value):
        """ Seed 

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("set seed to {}".format(str(seed_value)))


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
