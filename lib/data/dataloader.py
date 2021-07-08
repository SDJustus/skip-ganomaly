"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid, inference):
        self.train = train
        self.valid = valid
        self.inference = inference

##

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        path = os.path.split(path)[1]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

    
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """
    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    
    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    train_dl = None
    valid_dl = None
    inference_dl = None
    if opt.phase == "inference":
        inference_ds = ImageFolderWithPaths(os.path.join(opt.dataroot, 'inference'), transform)
        
        inference_dl = DataLoader(dataset=inference_ds, batch_size=1, shuffle=True, drop_last=False)
        
    else:
        train_ds = ImageFolderWithPaths(os.path.join(opt.dataroot, 'train'), transform)
        valid_ds = ImageFolderWithPaths(os.path.join(opt.dataroot, 'test'), transform)

        ## DATALOADER
        train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
        valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=True, drop_last=False)
    
    

    return Data(train_dl, valid_dl, inference_dl)
