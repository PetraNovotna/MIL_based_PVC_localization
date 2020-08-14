import torch
import numpy as np


class Log:
    def __init__(self):
        self.t = None

        self.model_names = []

        self.trainig_loss_log = []
        self.trainig_beta_log = []
        self.valid_loss_log = []
        self.valid_beta_log = []

        self.lbls_np_log = []
        self.res_np_log = []
        self.tmp_loss_log = []

    def save_tmp_log(self, lbls, res, loss):
        ## save values - epoch mean will be calucated later
        self.lbls_np_log.append(lbls.detach().cpu().numpy())
        self.res_np_log.append(res.detach().cpu().numpy())
        self.tmp_loss_log.append(loss.detach().cpu().numpy())

    def save_log_data_and_clear_tmp(self, train_or_test):

        lbls_np_log = np.concatenate(self.lbls_np_log, axis=0)
        res_np_log = np.concatenate(self.res_np_log, axis=0)

        self.t = 0.5
        ## get beta score - challage metric
        Fbeta, Gbeta, geom_mean = compute_beta_score(lbls_np_log, res_np_log > self.t, 2, 9)

        ## save appchoch averages
        if train_or_test == 'train':
            self.trainig_loss_log.append(np.mean(self.tmp_loss_log))
            self.trainig_beta_log.append(geom_mean)
        elif train_or_test == 'valid':
            self.valid_loss_log.append(np.mean(self.tmp_loss_log))
            self.valid_beta_log.append(geom_mean)
        else:
            raise ValueError('train or test')

        ## crear tmp epoch data buffers
        self.lbls_np_log = []
        self.res_np_log = []
        self.tmp_loss_log = []

    def save_log_model_name(self, model_name):
        ## store model names
        self.model_names.append(model_name)


def compute_beta_score(labels, output, beta=2, num_classes=9):
    TP = (labels == 1) & (output == 1)
    FP = (labels == 0) & (output == 1)
    FN = (labels == 1) & (output == 0)

    num_labels = np.sum(labels, axis=1, keepdims=1)
    if num_classes == 1:
        num_labels = 1

    num_classes = 1
    num_labels = 1

    TP = TP / num_labels
    FP = FP / num_labels
    FN = FN / num_labels

    TP = np.sum(TP, axis=0) / num_labels
    FP = np.sum(FP, axis=0) / num_labels
    FN = np.sum(FN, axis=0) / num_labels

    Fbetas = (1 + beta ** 2) * TP / ((1 + beta ** 2) * TP + FP + beta ** 2 * FN)

    Fbetas[((1 + beta ** 2) * TP + FP + beta ** 2 * FN) == 0] = 1

    Fbeta = np.mean(Fbetas)

    Fbeta = np.mean(Fbetas)

    Gbetas = TP / (TP + FP + beta * FN)

    Gbetas[(TP + FP + beta * FN) == 0] = 1

    Gbeta = np.mean(Gbetas)
    Gbeta = np.mean(Gbetas)

    geom_mean = np.sqrt(Gbeta * Fbeta)

    return Fbeta, Gbeta, geom_mean


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def wce(res, lbls, w_positive_tensor, w_negative_tensor):
    ## weighted crossetropy - weigths are for positive and negative
    res_c = torch.clamp(res, min=1e-6, max=1 - 1e-6)

    p1 = lbls * torch.log(res_c) * w_positive_tensor
    p2 = (1 - lbls) * torch.log(1 - res_c) * w_negative_tensor

    return -torch.mean(p1 + p2)