import os
import scipy.io as io
import torch
from torch import optim
from torch.utils import data
import matplotlib as plt
import numpy as np

from load_fncs import read_data
from load_fncs import read_lbl
from config import Config
from net import Net_addition_grow
import load_fncs as lf
from dataloader import Dataset
from logg_and_losses import get_lr
import net
from logg_and_losses import wce, Log



def train(names_train, names_valid):
    device = torch.device("cuda:0")

    ## ceate weigths for our loss - pathologies are rare so they need larger weigths
    lbl_counts = np.load(Config.info_save_dir + os.sep + 'lbl_counts.npy')
    num_of_sigs = np.load(Config.info_save_dir + os.sep + 'num_of_sigs.npy')
    w_positive = num_of_sigs / lbl_counts
    w_negative = num_of_sigs / (num_of_sigs - lbl_counts)
    w_positive_tensor = torch.from_numpy(w_positive.astype(np.float32)).to(device)
    w_negative_tensor = torch.from_numpy(w_negative.astype(np.float32)).to(device)

    training_generator = Dataset(names_train, Config.DATA_TMP_PATH, 'train')
    training_generator = data.DataLoader(training_generator, batch_size=Config.train_batch_size,
                                         num_workers=Config.train_num_workers, shuffle=True, drop_last=True,
                                         collate_fn=Dataset.collate_fn)

    validation_generator = Dataset(names_valid, Config.DATA_TMP_PATH, 'valid')
    validation_generator = data.DataLoader(validation_generator, batch_size=Config.valid_batch_size,
                                           num_workers=Config.valid_num_workers, shuffle=True, drop_last=False,
                                           collate_fn=Dataset.collate_fn)

    model = net.Net_addition_grow(levels=Config.levels, lvl1_size=Config.lvl1_size, input_size=Config.input_size,
                              output_size=Config.output_size,
                              convs_in_layer=Config.convs_in_layer, init_conv=Config.init_conv,
                              filter_size=Config.filter_size)

    model = model.to(device)

    ## create optimizer and learning rate scheduler to change learnng rate after
    optimizer = optim.Adam(model.parameters(), lr=Config.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, Config.step_size, gamma=Config.gamma, last_epoch=-1)

    ## select loss function
    loss_fcn = wce

    ## create empty log - object to save training results
    log = Log()

    for epoch in range(Config.max_epochs):

        # change model to training mode
        model.train()
        for pad_seqs, lens, lbls in training_generator:
            ## send data to graphic card
            pad_seqs, lens, lbls = pad_seqs.to(device), lens.to(device), lbls.to(device)

            ## aply model
            res, heatmap, score = model(pad_seqs, lens)

            ## calculate loss
            loss = loss_fcn(res, lbls, w_positive_tensor, w_negative_tensor)

            ## update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## save results
            log.save_tmp_log(lbls, res, loss)

        ## take mean of all batches -> get training performacne
        log.save_log_data_and_clear_tmp('train')

        ## validation mode - "disable" batch norm
        model.eval()
        for pad_seqs, lens, lbls in validation_generator:
            pad_seqs, lens, lbls = pad_seqs.to(device), lens.to(device), lbls.to(device)

            res, heatmap, score = model(pad_seqs, lens)

            heatmap0_np = heatmap.detach().cpu().numpy()[0, 0, :]
            pad_seqs0_np = pad_seqs.detach().cpu().numpy()[0, 1, :]

            len_short = int(lens.detach().cpu().numpy()[0])
            len_ratio = len_short / len(pad_seqs0_np)
            heatmap_end = int(round(len(heatmap0_np) * len_ratio))

            plt.figure(figsize=(15, 7))
            plt.plot(heatmap0_np[0:heatmap_end], 'b')
            plt.title(str(res.detach().cpu().numpy()[0, 0]))
            plt.show()
            plt.figure(figsize=(15, 7))
            plt.plot(pad_seqs0_np[0:len_short], 'r')
            plt.title(str(lbls.detach().cpu().numpy()[0]))
            plt.show()

            loss = loss_fcn(res, lbls, w_positive_tensor, w_negative_tensor)

            log.save_tmp_log(lbls, res, loss)

        log.save_log_data_and_clear_tmp('valid')

        lr = get_lr(optimizer)

        info = str(epoch) + '_' + str(lr) + '_train_' + str(log.trainig_beta_log[-1]) + '_valid_' + str(
            log.valid_beta_log[-1])
        print(info)

        model_name = Config.model_save_dir + os.sep + Config.model_note + info + '.pkl'
        log.save_log_model_name(model_name)
        model.save_log(log)
        model.save_config(Config)
        torch.save(model, model_name)

        ## plot loss and beta score
        model.plot_training()

        scheduler.step()

    return log


split = np.load(Config.info_save_dir + os.sep + 'split.npy', allow_pickle=True).item()

try:
    os.mkdir(Config.best_models_dir)
except:
    pass

try:
    os.mkdir(Config.model_save_dir)
except:
    pass

log = train(split['train'], split['valid'])