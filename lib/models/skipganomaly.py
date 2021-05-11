"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
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

from lib.models.networks import NetD, weights_init, define_G, define_D, get_scheduler
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import roc
from lib.models.basemodel import BaseModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
#import wandb



class Skipganomaly(BaseModel):
    """GANomaly Class
    """
    @property
    def name(self): return 'skip-ganomaly'

    def __init__(self, opt, data=None):
        super(Skipganomaly, self).__init__(opt, data)
        ##

        # -- Misc attributes
        self.add_noise = True
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = define_G(self.opt, norm='batch', use_dropout=False, init_type='normal')
        self.netd = define_D(self.opt, norm='batch', use_sigmoid=False, init_type='normal')

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        if self.opt.verbose:
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
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizers = []
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        else:
            self.real_images = []
    def forward(self):
        self.forward_g()
        self.forward_d()

    def forward_g(self):
        """ Forward propagate through netG
        """
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
        self.err_g_lat = self.opt.w_lat * self.l_lat(self.feat_fake, self.feat_real)

        self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat
        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        # Fake
        pred_fake, _ = self.netd(self.fake.detach())
        self.err_d_fake = self.l_adv(pred_fake, self.fake_label)

        # Real
        # pred_real, feat_real = self.netd(self.input)
        self.err_d_real = self.l_adv(self.pred_real, self.real_label)

        # Combine losses.
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


            scores = {}

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.features = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

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
                self.set_input(data)
                self.fake = self.netg(self.input)

                real_clas, self.feat_real = self.netd(self.input)
                fake_clas, self.feat_fake = self.netd(self.fake)

                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9*rec + 0.1*lat

                time_o = time.time()

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    #iterate over them (real) and write anomaly score and ground truth on filename
                    vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True)

                if self.opt.isTrain is False:
                    self.real_images.extend(deepcopy(real))
                i = i + 1
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                             (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc, threshold = roc(self.gt_labels, self.an_scores, output_directory=self.opt.outf, epoch=self.epoch)


            

            ##
            # PLOT PERFORMANCE
            if self.opt.display and self.opt.phase == 'test':
                plt.ion()
                # Create data frame for scores and labels.
                scores["scores"] = self.an_scores.cpu()
                scores["labels"] = self.gt_labels.cpu()
                print("Scores: " + str(scores["scores"]))
                
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv(self.opt.outf + "/histogram" + str(self.epoch) + ".csv")

                # Filter normal and abnormal scores.
                abn_scr = hist.loc[hist.labels == 1]['scores']
                nrm_scr = hist.loc[hist.labels == 0]['scores']

                # Create figure and plot the distribution.
                fig, axis = plt.subplots(figsize=(4,4))
                sns.distplot(nrm_scr, label=r'Normal Scores', ax=axis)
                sns.distplot(abn_scr, label=r'Abnormal Scores', ax=axis)
                axis.vlines

                plt.legend()
                plt.yticks([])
                plt.xlabel(r'Anomaly Scores')
                self.visualizer.writer.add_figure("Histogram", fig, self.epoch)
                
                #self.visualizer.writer.add_custom_scalars_multilinechart()

            scores["scores"][scores["scores"] >= threshold] = 1
            scores["scores"][scores["scores"] < threshold] = 0
            #print(scores)
            precision, recall, f1_score, support = precision_recall_fscore_support(scores["labels"], scores["scores"],
                                                                                   average="binary", pos_label=0)
            # conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]
            conf_matrix = confusion_matrix(scores["labels"], scores["scores"])
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc), ('precision', precision),
                                       ("recall", recall), ("F1_Score", f1_score), ("conf_matrix", conf_matrix),
                                       ("threshold", threshold)
                                          #, ("support", support)
                                        # , ("conf_matrix", conf_matrix)
                                       ])
            if self.opt.display and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.writer.add_pr_curve("pr_curve", scores["labels"], scores["scores"])
                
                self.visualizer.plot_current_conf_matrix(self.epoch, performance["conf_matrix"])
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            if self.opt.isTrain is False:
                i = 0

                for image, gt, anomaly_score in zip(self.real_images, scores["labels"].numpy(), scores["scores"].numpy()):
                    anomaly_score=int(anomaly_score)
                    name = ""
                    if gt == anomaly_score == 0:
                        name = "tp"
                    if anomaly_score == 0 and gt != anomaly_score:
                        name = "fp"
                    if gt == anomaly_score == 1:
                        name = "tn"
                    if anomaly_score == 1 and gt != anomaly_score:
                        name = "fn"
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'inference').replace("\\","/")
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    vutils.save_image(image, '{0}/{1}_{2}.png'.format(str(dst), str(i), name), normalize=True)
                    i = i + 1





            ##
            # PLOT HISTOGRAM
            if True:
                plt.ion()
                # Create data frame for scores and labels.
                scores["scores"] = self.an_scores.cpu()
                scores["labels"] = self.gt_labels.cpu()
                print("Scores: " + str(scores["scores"]))
                
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv(self.opt.outf + "/histogram" + str(self.epoch) + ".csv")

                # Filter normal and abnormal scores.
                abn_scr = hist.loc[hist.labels == 1]['scores']
                nrm_scr = hist.loc[hist.labels == 0]['scores']

                # Create figure and plot the distribution.
                fig, axis = plt.subplots(figsize=(4,4))
                sns.distplot(nrm_scr, label=r'Normal Scores', ax=axis)
                sns.distplot(abn_scr, label=r'Abnormal Scores', ax=axis)
                

                plt.legend()
                plt.yticks([])
                plt.xlabel(r'Anomaly Scores')

            ##
            # PLOT PERFORMANCE
            if self.opt.display and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.writer.add_pr_curve("pr_curve", scores["labels"], scores["scores"])
                self.visualizer.writer.add_figure("Histogram", fig, self.epoch)
                self.visualizer.plot_current_conf_matrix(self.epoch, performance["conf_matrix"])
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
                #self.visualizer.writer.add_custom_scalars_multilinechart()

            ##
            # RETURN
            return performance
