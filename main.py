import os
import scipy.io as io
import torch
from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch.nn.functional as F

from utils.load_fncs import read_data
from utils.load_fncs import read_lbl
from config import Config
from net import Net_addition_grow
import utils.load_fncs as lf
from dataloader import Dataset
from utils.losses import get_lr, wce
import net
from utils.get_stats import get_stats
from utils.adjustLearningRateAndLoss import AdjustLearningRateAndLoss
from utils.log import Log




def train():
    device = torch.device("cuda:0")

    
    
    # file_list = glob.glob(Config.DATA_TMP_PATH + "/*.mat")
    
    file_list = glob.glob(Config.lbls_path + "/*.mat")
    file_list = [x.replace('_naklikane.mat','.mat').replace(Config.lbls_path,Config.DATA_TMP_PATH) for x in file_list]
    
    num_of_sigs=len(file_list)
    
    if Config.is_mil:
        lbl_counts,lenss=get_stats(file_list)
        w_positive = num_of_sigs / lbl_counts
        w_negative = num_of_sigs / (num_of_sigs - lbl_counts)
        w_positive_tensor = torch.from_numpy(w_positive.astype(np.float32)).to(device)
        w_negative_tensor = torch.from_numpy(w_negative.astype(np.float32)).to(device)


    state=np.random.get_state()
    np.random.seed(42)
    split_ratio_ind = int(np.floor(Config.SPLIT_RATIO[0] / (Config.SPLIT_RATIO[0] + Config.SPLIT_RATIO[1]) * num_of_sigs))
    permuted_idx = np.random.permutation(num_of_sigs)
    train_ind = permuted_idx[:split_ratio_ind]
    valid_ind = permuted_idx[split_ratio_ind:]
    partition = {"train": [file_list[file_idx] for file_idx in train_ind],
                 "valid": [file_list[file_idx] for file_idx in valid_ind]}


    training_generator = Dataset( partition["train"], 'train')
    training_generator = data.DataLoader(training_generator, batch_size=Config.train_batch_size,
                                         num_workers=Config.train_num_workers, shuffle=True, drop_last=True,
                                         collate_fn=Dataset.collate_fn)

    validation_generator = Dataset(partition["valid"], 'valid')
    validation_generator = data.DataLoader(validation_generator, batch_size=Config.valid_batch_size,
                                           num_workers=Config.valid_num_workers, shuffle=False, drop_last=False,
                                           collate_fn=Dataset.collate_fn)


    model = net.Net_addition_grow(levels=Config.levels, lvl1_size=Config.lvl1_size, input_size=Config.input_size,
                              output_size=Config.output_size,
                              convs_in_layer=Config.convs_in_layer, init_conv=Config.init_conv,
                              filter_size=Config.filter_size)

    model = model.to(device)

    ## create optimizer and learning rate scheduler to change learnng rate after
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR_LIST[0], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    scheduler=AdjustLearningRateAndLoss(optimizer,Config.LR_LIST,Config.LR_CHANGES_LIST,Config.LOSS_FUNTIONS)


    ## create empty log - object to save training results
    log = Log(['loss'])

    for epoch in range(Config.max_epochs):

        N=len(training_generator)
        # change model to training mode
        model.train()
        for it,(pad_seqs, lens, lbls,file_names,detection)  in enumerate(training_generator):
            
            if it%3==0:
                print(str(it) + '/' + str(N))
                
            ## send data to graphic card
            pad_seqs, lens, lbls,detection  = pad_seqs.to(device), lens.to(device), lbls.to(device), detection.to(device)

            ## aply model
            res, heatmap, score,detection_subsampled = model(pad_seqs, lens,detection)

            
            ## calculate loss
            if Config.is_mil:
                loss=scheduler.actual_loss(res, lbls, w_positive_tensor, w_negative_tensor)
            else:
                loss=scheduler.actual_loss(heatmap, detection_subsampled)

            ## update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            ## save results
            log.append_train([loss])



        N=len(validation_generator)
        ## validation mode - "disable" batch norm
        model.eval()
        for it,(pad_seqs, lens, lbls,file_names,detection)  in enumerate(validation_generator):
            
            if it%3==0:
                print(str(it) + '/' + str(N))
            
            
            pad_seqs, lens, lbls,detection  = pad_seqs.to(device), lens.to(device), lbls.to(device), detection.to(device)

            res, heatmap, score,detection_subsampled = model(pad_seqs, lens,detection)

            
            ## calculate loss
            if Config.is_mil:
                loss=scheduler.actual_loss(res, lbls, w_positive_tensor, w_negative_tensor)
            else:
                loss=scheduler.actual_loss(heatmap, detection_subsampled)

            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()
            detection_subsampled=detection_subsampled.detach().cpu().numpy()
            heatmap=heatmap.detach().cpu().numpy()

            ## save results
            log.append_test([loss])


            heatmap0_np = heatmap[0, 0, :]
            pad_seqs0_np = pad_seqs.detach().cpu().numpy()[0, 1, :]

            len_short = int(lens.detach().cpu().numpy()[0])
            len_ratio = len_short / len(pad_seqs0_np)
            heatmap_end = int(round(len(heatmap0_np) * len_ratio))
            

            if epoch == (Config.max_epochs-1):
                heatmap_np = heatmap
                pad_seq_np = pad_seqs.detach().cpu().numpy()
                lens_np = lens.detach().cpu().numpy()

                for hm_index in range(heatmap_np.shape[0]):
                    heatmap0_np = heatmap_np[hm_index, :, :]
                    pad_seqs0_np = pad_seq_np[hm_index,:,:]


                    len_short = int(lens_np[hm_index])
                    len_ratio = len_short / pad_seqs0_np.shape[1]
                    heatmap_end = int(round(heatmap0_np.shape[1] * len_ratio))
                    heatmap0_np = heatmap0_np[:,0:heatmap_end]
                    N=heatmap0_np.shape[1]
                    heatmap0_np_res=[]
                    for k in range(heatmap0_np.shape[0]):
                        heatmap0_np_res.append(np.interp(np.linspace(0, N - 1, len_short),np.linspace(0, N - 1, N), heatmap0_np[k,:]))
                    heatmap0_np_res=np.stack(heatmap0_np_res,0)


                    np.save(Config.res_dir + os.sep + file_names[hm_index]+"_heatmap.npy",heatmap0_np_res)


        plt.plot(heatmap[0,0,:])
        plt.plot(detection_subsampled[0,0,:])
        plt.show()
        
        
        log.save_and_reset()

        lr = get_lr(optimizer)

        info = str(epoch) + '_' + str(lr) + '_train_' + str(log.train_log['loss'][-1]) + '_valid_' + str(log.test_log['loss'][-1])
        print(info)

        model_name = Config.model_save_dir + os.sep + Config.model_note + info + '.pt'
        log.save_log_model_name(model_name)
        model.save_log(log)
        model.save_config(Config)
        torch.save(model, model_name)

        log.plot(model_name)
        
        scheduler.step()



if __name__ == "__main__":
    
    train()