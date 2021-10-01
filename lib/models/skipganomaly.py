"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
from re import I
import time
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from lib.models.networks import weights_init, define_G, define_D, get_scheduler
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import roc, auprc, write_inference_result, get_performance
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
#import wandb


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
class Skipganomaly:
    """Skip-GANomaly Class
    """
    @property
    def name(self): return 'skip-ganomaly'

    def __init__(self, opt, data=None):
        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.data = data
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.inf_dir = os.path.join(self.opt.outf, self.opt.name, 'inference')
        self.device = torch.device("cuda:0" if self.opt.device != "cpu" else "cpu")

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks from networks.py.
        
        self.netg = define_G(self.opt, norm='batch', use_dropout=False, init_type='normal')
        self.netd = define_D(self.opt, norm='batch', use_sigmoid=False, init_type='normal')

        ##
        #resume Training 
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        print(self.netg)
        print(self.netd)

        ##
        # Loss Functions
        self.l_adv = nn.BCELoss()
        self.l_con = nn.L1Loss()
        self.l_lat = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.noise = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.phase == "train":
            self.netg.train()
            self.netd.train()
            self.optimizers = []
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

            
    ##
    def set_input(self, input:torch.Tensor, noise:bool=False):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Add noise to the input.
            if noise: self.noise.data.copy_(torch.randn(self.noise.size()))

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([
            ('err_d', self.err_d),
            ('err_g', self.err_g),
            ('err_g_adv', self.err_g_adv),
            ('err_g_con', self.err_g_con),
            ('err_g_lat', self.err_g_lat)])

        return errors

    ##
    def reinit_d(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('Reloading d net')

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch:int, is_best:bool=False):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """
        name = self.opt.dataset if self.opt.dataset else self.opt.dataroot.split("/")[-1]
        weight_dir = os.path.join(
            self.opt.outf, name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if is_best:
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f'{weight_dir}/netG_best.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f'{weight_dir}/netD_best.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f"{weight_dir}/netD_{epoch}.pth")
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f"{weight_dir}/netG_{epoch}.pth")

    def load_weights(self, epoch=None, is_best:bool=False, path=None):
        """ Load pre-trained weights of NetG and NetD

        Keyword Arguments:
            epoch {int}     -- Epoch to be loaded  (default: {None})
            is_best {bool}  -- Load the best epoch (default: {False})
            path {str}      -- Path to weight file (default: {None})

        Raises:
            Exception -- [description]
            IOError -- [description]
        """

        if epoch is None and is_best is False:
            raise Exception('Please provide epoch to be loaded or choose the best epoch.')

        if is_best:
            fname_g = f"netG_best.pth"
            fname_d = f"netD_best.pth"
        else:
            fname_g = f"netG_{epoch}.pth"
            fname_d = f"netD_{epoch}.pth"

        if path is None:
            name = self.opt.dataset if self.opt.dataset else self.opt.dataroot.split("/")[-1]
            path_g = f"{self.opt.outf}/{name}/train/weights/{fname_g}"
            path_d = f"{self.opt.outf}/{name}/train/weights/{fname_d}"

        else:
            path_g = path + "/" + fname_g
            path_d = path + "/" + fname_d

        # Load the weights of netg and netd.
        print('>> Loading weights...')

        if len(self.opt.gpu_ids) == 0:
            weights_g = torch.load(path_g, map_location=lambda storage, loc: storage)['state_dict']
            weights_d = torch.load(path_d, map_location=lambda storage, loc: storage)['state_dict']
        else:
            weights_g = torch.load(path_g)['state_dict']
            weights_d = torch.load(path_d)['state_dict']
        try:
            # create new OrderedDict that does not contain `module.`

            new_weights_g = OrderedDict()
            new_weights_d = OrderedDict()
            for k, v in weights_g.items():
                name = k[7:]  # remove `module.`
                new_weights_g[name] = v
            for k, v in weights_d.items():
                name = k[7:]  # remove `module.`
                new_weights_d[name] = v
            # load params
            if len(self.opt.gpu_ids) == 0:
                weights_g = new_weights_g
                weights_d = new_weights_d
            self.netg.load_state_dict(weights_g)
            self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("netG weights not found")
        print('   Done.')


    def forward(self):
        self.forward_g()
        self.forward_d()

    def forward_g(self):
        """ Forward propagate through netG
        """
        #TODO: Check, why noised input is used
        self.fake = self.netg(self.input + self.noise)

    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake)

    def backward_g(self):
        """ Backpropagate netg
        """
        self.err_g_adv = self.opt.w_adv * self.l_adv(self.pred_fake, self.real_label)
        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input)
        self.err_g_lat = self.opt.w_lat * self.l_lat(self.feat_fake, self.feat_real) # should be named discriminator
        if self.opt.verbose:
            print(f'err_g_adv: {str(self.err_g_adv)}')
            print(f'err_g_con: {str(self.err_g_con)}')
            print(f'err_g_lat: {str(self.err_g_lat)}')

        self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat
        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        # Fake
        #print(f'pref_fake: {str(self.pred_fake)}')
        #print(f'self.fake_label: {str(self.fake_label)}')
        #print(f'self.pred_real: {str(self.pred_real)}')
        #print(f'self.real_label: {str(self.real_label)}')
        self.err_d_fake = self.l_adv(self.pred_fake, self.fake_label)

        # Real
        # pred_real, feat_real = self.netd(self.input)
        self.err_d_real = self.l_adv(self.pred_real, self.real_label)

        # Combine losses.
        # TODO: According to https://github.com/samet-akcay/skip-ganomaly/issues/18#issue-728932038 ... Check if lat loss has to be negative in discriminator backprob
        if self.opt.verbose:
            print(f'err_d_real: {str(self.err_d_real)}')
            print(f'err_d_fake: {str(self.err_d_fake)}')
            print(f'err_g_lat: {str(self.err_g_lat)}')
        self.err_d = self.err_d_real + self.err_d_fake + self.err_g_lat
        self.err_d.backward(retain_graph=True)

    def update_netg(self):
        """ Update Generator Network.
        """       
        self.optimizer_g.zero_grad()
        self.backward_g()

    def update_netd(self):
        """ Update Discriminator Network.
        """       
        self.optimizer_d.zero_grad()
        self.backward_d()
        
    ##
    def optimize_params(self):
        """ Optimize netD and netG  networks.
        """
        self.forward()
        self.update_netg()
        self.update_netd()
        self.optimizer_g.step()
        self.optimizer_d.step()
        if self.err_d < 1e-5: self.reinit_d()
        
    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """
        self.opt.phase = "train"
        self.netg.train()
        self.netd.train()
        epoch_iter = 0
        for data in tqdm(self.data.train, leave=False, total=len(self.data.train)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize
            

            self.set_input(data)
            self.optimize_params()
            reals, fakes, fixed = self.get_current_images()
                
            errors = self.get_errors()
            
            if self.opt.display:                    
                self.visualizer.plot_current_errors(self.epoch, self.total_steps, errors)
                # Write images to tensorboard
                if self.total_steps % self.opt.save_image_freq == 0:
                    self.visualizer.display_current_images(reals, fakes, fixed, train_or_test="train", global_step=self.total_steps)
                

            if self.total_steps % self.opt.save_image_freq == 0:
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(f">> Training {self.name} on {self.opt.dataset} to detect anomalies")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            self.train_one_epoch()
            res = self.test()
            if res['auc'] > best_auc:
                best_auc = res['auc']
                self.save_weights(self.epoch, is_best=True)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self, plot_hist=True):
        """ Test GANomaly model.

        Args:
            data ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        self.netg.eval()
        self.netd.eval()
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.path_to_weights is not None:
                if self.opt.epoch is None:
                    raise ValueError("Need value for epoch of the weights")
                self.load_weights(path=self.opt.path_to_weights, epoch=self.opt.epoch)

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)

            print("   Testing %s" % self.name)
            self.times = []
            total_steps_test = 0
            epoch_iter = 0
            i = 0
            for data in tqdm(self.data.valid, leave=False, total=len(self.data.valid)):
                total_steps_test += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()

                # Forward - Pass
                self.forward_for_testing(data)

                # Calculate the anomaly score.
                error = self.calculate_an_score()

                time_o = time.time()

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))
                if self.opt.verbose:
                    print(f'an_scores: {str(self.an_scores)}')
                self.times.append(time_o - time_i)
                real, fake, fixed = self.get_current_images()
                
                if self.epoch*len(self.data.valid)+total_steps_test % self.opt.save_image_freq == 0:
                    self.visualizer.display_current_images(real, fake, fixed, train_or_test="test", global_step=self.epoch*len(self.data.valid)+total_steps_test)
                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    
                    #iterate over them (real) and write anomaly score and ground truth on filename
                    vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True)
                i = i + 1
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times * 1000)

            # Scale error vector between [0, 1]
            # self.an_scores = (self.an_scores - torch.min(self.an_scores))/(torch.max(self.an_scores) - torch.min(self.an_scores))
            if self.opt.verbose:
                print(f'scaled an_scores: {str(self.an_scores)}')
            y_trues = self.gt_labels.cpu()
            y_preds = self.an_scores.cpu()
                # Create data frame for scores and labels.
            performance, thresholds, _ = get_performance(y_trues=y_trues, y_preds=y_preds, manual_threshold=self.opt.decision_threshold)
            with open(os.path.join(self.opt.outf, self.opt.phase +"_results.txt"), "a+") as f:
                f.write(str(performance))
                f.write("\n")
                f.close()
            self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["threshold"], save_path=os.path.join(self.opt.outf, "histogram_test"+str(self.epoch)+".png"), tag="Histogram_Test", global_step=self.epoch)
            self.visualizer.plot_pr_curve(y_trues=y_trues, y_preds=y_preds, thresholds=thresholds, global_step=self.epoch, tag="PR_Curve_Test")
            self.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=self.epoch, tag="ROC_Curve_Test")
                        
            self.visualizer.plot_current_conf_matrix(self.epoch, performance["conf_matrix"], tag="Confusion_Matrix_Test")
            
            self.visualizer.plot_performance(self.epoch, 0, performance, tag="Performance_Test")
            return performance

    def forward_for_testing(self, data):
        self.set_input(data)
        self.fake = self.netg(self.input)

        real_clas, self.feat_real = self.netd(self.input)
        fake_clas, self.feat_fake = self.netd(self.fake)

    
    
    def inference(self):
        self.netg.eval()
        self.netd.eval()
        with torch.no_grad():
            self.load_weights(path=self.opt.path_to_weights, is_best=True)
            

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.inference.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.inference.dataset),), dtype=torch.long, device=self.device)
            
            print("Starting Inference!")
            inf_time = None
            inf_times = []
            
            self.file_names = []
            for i, data in tqdm(enumerate(self.data.inference), leave=False, total=len(self.data.inference)):
                inf_start = time.time()

                # Forward - Pass
                self.forward_for_testing(data)

                # Calculate the anomaly score.
                error = self.calculate_an_score()
                
                inf_times.append(time.time()-inf_start)

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))
                if self.opt.verbose:
                    print(f'an_scores: {str(self.an_scores)}')
                real, fake, fixed = self.get_current_images()
                
                if i % self.opt.save_image_freq == 0:
                    self.visualizer.display_current_images(real, fake, fixed, train_or_test="test_inference", global_step=i)
                
                self.file_names.append(data[2])
            # Measure inference time.
            
                

        # Scale error vector between [0, 1] TODO: does it work without normalizing?
        # self.an_scores = (self.an_scores - torch.min(self.an_scores))/(torch.max(self.an_scores) - torch.min(self.an_scores))
        if self.opt.verbose:
            print(f'scaled an_scores: {str(self.an_scores)}')
        y_trues = self.gt_labels.cpu()
        y_preds = self.an_scores.cpu()
        inf_time = sum(inf_times)
        print (f'Inference time: {inf_time} secs')
        print (f'Inference time / individual: {inf_time/len(y_trues)} secs')
        
            # Create data frame for scores and labels.
        performance, thresholds, y_preds_after_threshold = get_performance(y_trues=y_trues, y_preds=y_preds, manual_threshold=self.opt.decision_threshold)
        with open(os.path.join(self.opt.outf, self.opt.phase +"_results.txt"), "w") as f:
            f.write(str(performance))
            f.close()
        self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["threshold"], save_path=os.path.join(self.opt.outf, "histogram_inference.png"), tag="Histogram_Inference")
        self.visualizer.plot_pr_curve(y_trues=y_trues, y_preds=y_preds, thresholds=thresholds, global_step=1, tag="PR_Curve_Inference")
        self.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=1, tag="ROC_Curve_Inference")
        
        self.visualizer.plot_current_conf_matrix(1, performance["conf_matrix"], tag="Confusion_Matrix_Inference")
        if self.opt.decision_threshold:
            self.visualizer.plot_current_conf_matrix(2, performance["conf_matrix_man"])
            self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["manual_threshold"], global_step=2, save_path=os.path.join(self.opt.outf, "histogram_inference_man.png"), tag="Histogram_Inference")
        self.visualizer.plot_performance(1, 0, performance, tag="Performance_Inference")
        
            
        write_inference_result(file_names=self.file_names, y_trues=y_trues, y_preds=y_preds_after_threshold,outf=os.path.join(self.opt.outf, "classification_result.json"))
        ##
        # RETURN
        return performance

    def calculate_an_score(self):
        si = self.input.size()
        sz = self.feat_real.size()
        rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
        lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
        rec = torch.mean(torch.pow(rec, 2), dim=1)
        lat = torch.mean(torch.pow(lat, 2), dim=1)
        #print("lat", lat)
        #print("rec", rec)
        if self.opt.verbose:
            print(f'rec: {str(rec)}')
            print(f'lat: {str(lat)}')
        error = 0.9*rec + 0.1*lat
        
        return error