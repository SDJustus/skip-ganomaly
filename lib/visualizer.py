""" This file contains Visualizer class based on Facebook's visdom.

Returns:
    Visualizer(): Visualizer class to display plots and images
"""

##
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torchvision.utils as vutils
from .plot import plot_confusion_matrix
from .evaluate import get_values_for_pr_curve, get_values_for_roc_curve
import seaborn as sns

##
class Visualizer():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """
    # pylint: disable=too-many-instance-attributes
    # Reasonable.

    ##
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.writer = None
        # use tensorboard for now
        if self.opt.display:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join("../tensorboard/skip_ganomaly/", opt.outf))

        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)
        now  = time.strftime("%c")
        title = f'================ {now} ================\n'
        info  = f'Anomalies, {opt.nz}, {opt.w_adv}, {opt.w_con}, {opt.w_lat}\n'
        self.write_to_log_file(text=title + info)


    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##
    def plot_current_errors(self, epoch, total_steps, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        self.writer.add_scalars("Loss over time", errors, global_step=total_steps)
        

    ##
    def plot_performance(self, epoch, counter_ratio, performance, tag=None):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        
        self.writer.add_scalars(tag if tag else "Performance Metrics", {k:v for k,v in performance.items() if ("conf_matrix" not in k and k != "Avg Run Time (ms/batch)")}, global_step=epoch)
            
        
    def plot_current_conf_matrix(self, epoch, cm, tag=None):
        plot = plot_confusion_matrix(cm, normalize=False, save_path=os.path.join(self.opt.outf, self.opt.phase+"_conf_matrix.png"))
        self.writer.add_figure(tag if tag else "Confusion Matrix", plot, global_step=epoch)
        

    ##
    def print_current_errors(self, epoch, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
            batch_i (int): Current batch
            batch_n (int): Total Number of batches.
        """
        # message = '   [%d/%d] ' % (epoch, self.opt.niter)
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    ##
    def write_to_log_file(self, text):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % text)

    ##
    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        #print(performance)
        for key, val in performance.items():
            if key == "conf_matrix":
                message += '%s: %s ' % (key, val)
            else:
                message += '%s: %.3f ' % (key, val)
        message += 'max AUC: %.3f' % best

        print(message)
        self.write_to_log_file(text=message)

    def display_current_images(self, reals, fakes, fixed, train_or_test="train", global_step=0):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        # fixed = self.normalize(fixed.cpu().numpy())
        self.writer.add_images("Reals from {} step: ".format(str(train_or_test)), reals, global_step=global_step)
        self.writer.add_images("Fakes from {} step: ".format(str(train_or_test)), fakes, global_step=global_step)
        
    def plot_pr_curve(self, y_trues, y_preds, thresholds, global_step, tag=None):
        tp_counts, fp_counts, tn_counts, fn_counts, precisions, recalls, n_thresholds = get_values_for_pr_curve(y_trues, y_preds, thresholds)
        self.writer.add_pr_curve_raw(tag if tag else "Precision_recall_curve", true_positive_counts=tp_counts, false_positive_counts=fp_counts, true_negative_counts=tn_counts, false_negative_counts= fn_counts,
                                             precision=precisions, recall=recalls, num_thresholds=n_thresholds, global_step=global_step)
        
    def save_current_images(self, epoch, reals, fakes, fixed):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        vutils.save_image(reals, '%s/reals.png' % self.img_dir, normalize=True)
        vutils.save_image(fakes, '%s/fakes.png' % self.img_dir, normalize=True)
        vutils.save_image(fixed, '%s/fixed_fakes_%03d.png' %(self.img_dir, epoch+1), normalize=True)
        
    def plot_histogram(self, y_trues, y_preds, threshold, global_step=1, save_path=None, tag=None):
        scores = dict()
        scores["scores"] = y_preds
        scores["labels"] = y_trues
        hist = pd.DataFrame.from_dict(scores)
        
        plt.ion()

            # Filter normal and abnormal scores.
        abn_scr = hist.loc[hist.labels == 1]['scores']
        nrm_scr = hist.loc[hist.labels == 0]['scores']

            # Create figure and plot the distribution.
        fig = plt.figure(figsize=(4,4))
        sns.distplot(nrm_scr, label=r'Normal Scores')
        sns.distplot(abn_scr, label=r'Abnormal Scores')
        plt.axvline(threshold, 0, 1, label='threshold', color="red")
        plt.legend()
        plt.yticks([])
        plt.xlabel(r'Anomaly Scores')
        plt.savefig(save_path)
        self.writer.add_figure(tag if tag else "Histogram", fig, global_step)

    def plot_roc_curve(self, y_trues, y_preds, global_step=1, tag=None, save_path=None):
        fpr, tpr, roc_auc = get_values_for_roc_curve(y_trues, y_preds)
        fig = plt.figure(figsize=(4,4))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)
        self.writer.add_figure(tag if tag else "ROC-Curve", fig, global_step)