from torch.utils import data
import numpy as np
import torch
from config import Config
import utils.load_fncs as lf
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class Dataset(data.Dataset):

    def __init__(self, data_names, split):
        """Initialization"""
        self.data_names = data_names
        self.split = split
        self.pato_names = Config.pato_names

    def __len__(self):
        """Return total number of data samples"""
        return len(self.data_names)

    def __getitem__(self, idx):
        """Generate data sample"""
        # Select sample
        file_name = self.data_names[idx]

        # Read data and get label
        X = lf.read_data(file_name)
            
        sig_len = X.shape[1]
        signal_num = X.shape[0]
        

        

        head,tail = os.path.split(file_name)
        lbl_file_name=Config.lbls_path + os.sep + tail.replace('.mat','_position_labels.mat')
        lbl_PAC,lbl_PVC = lf.read_lbl_pos(lbl_file_name)


        y  = np.zeros(2,dtype = np.float32)
        if len(lbl_PAC)>0:
            y[0] = 1
        if len(lbl_PVC)>0:
            y[1] = 1



        Y_PAC=np.zeros((sig_len))
        
        Y_PAC[lbl_PAC]=1
        
        Y_PAC = gaussian_filter(Y_PAC,Config.gaussian_sigma,mode='constant')/(1/(Config.gaussian_sigma*np.sqrt(2*np.pi)))
        Y_PAC = Y_PAC.reshape((1,len(Y_PAC))).astype(np.float32)
        
        
        
        Y_PVC=np.zeros((sig_len))
        
        Y_PVC[lbl_PVC]=1
        
        Y_PVC = gaussian_filter(Y_PVC,Config.gaussian_sigma,mode='constant')/(1/(Config.gaussian_sigma*np.sqrt(2*np.pi)))
        Y_PVC = Y_PVC.reshape((1,len(Y_PVC))).astype(np.float32)
        
        
        Y = np.concatenate((Y_PAC,Y_PVC),axis = 0)
        lbl_num = Y.shape[0]
        
        if Y.shape[0] ==1:
            pass
        

        
        ## normalization
        X = X  / 200


        ##augmentation
        if self.split == 'train':
            ##random circshift
            if torch.rand(1).numpy()[0] > 0.3:
                shift = torch.randint(sig_len, (1, 1)).view(-1).numpy()

                X = np.roll(X, shift, axis=1)
                Y = np.roll(Y, shift, axis=1)

            # # ranzomly inserted zeros
            # if torch.rand(1).numpy()[0] > 0.3:

            #     rand_pos = int(torch.randint(sig_len, (1, 1)).view(-1).numpy())
            #     if sig_len - rand_pos > 1000:
            #         X[:, rand_pos:rand_pos + 1000] = 0
            #     else:
            #         X[:, rand_pos:] = 0

            ## random stretch -
            if torch.rand(1).numpy()[0] > 0.3:

                max_resize_change = 0.2
                relative_change = 1 + torch.rand(1).numpy()[0] * 2 * max_resize_change - max_resize_change
                ##mutliply by 2 is same as equvalent to multiply by 0.5 not 0!
                if relative_change<1:
                    relative_change=1/(1-relative_change+1)
                
                new_len = int(relative_change * sig_len)

                XX = np.zeros((signal_num, new_len))
                for k in range(signal_num):
                    XX[k, :] = np.interp(np.linspace(0, sig_len - 1, new_len), np.linspace(0, sig_len - 1, sig_len),
                                        X[k, :])
                X = XX
                
                
                YY= np.zeros((lbl_num, new_len))
                for k in range(lbl_num):
                    YY[k, :] = np.interp(np.linspace(0, sig_len - 1, new_len), np.linspace(0, sig_len - 1, sig_len),
                                        Y[k, :])
                    
                Y=YY
                

            ## random multiplication of each lead by a number
            if torch.rand(1).numpy()[0] > 0.3:

                max_mult_change = 0.4

                for k in range(signal_num):
                    mult_change = 1 + torch.rand(1).numpy()[0] * 2 * max_mult_change - max_mult_change
                    ##mutliply by 2 is same as equvalent to multiply by 0.5 not 0!
                    if mult_change<1:
                        mult_change=1/(1-mult_change+1)
                        
                    X[k, :] = X[k, :] * mult_change



        return X, y,file_name,Y

    def collate_fn(data):
        ## this take list of samples and put them into batch

        ##pad with zeros
        pad_val = 0

        ## get list of singals and its lengths
        seqs, lbls, file_names,lbls_seqs = zip(*data)

        lens = [seq.shape[1] for seq in seqs]

        ## pad shorter signals with zeros to make them same length
        padded_seqs = pad_val * np.ones((len(seqs), seqs[0].shape[0], np.max(lens))).astype(np.float32)
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :, :end] = seq
            
        padded_lbls_seqs = pad_val * np.ones((len(lbls_seqs), lbls_seqs[0].shape[0], np.max(lens))).astype(np.float32)
        for i, seq in enumerate(lbls_seqs):
            end = lens[i]
            padded_lbls_seqs[i, :, :end] = seq    

        ## stack and reahape signal lengts to 10 vector
        lbls = np.stack(lbls, axis=0)
        lbls = lbls.reshape(lbls.shape[0:2])
        lens = np.array(lens).astype(np.float32)

        ## numpy -> torch tensor
        padded_seqs = torch.from_numpy(padded_seqs)
        lbls = torch.from_numpy(lbls)
        lens = torch.from_numpy(lens)
        padded_lbls_seqs = torch.from_numpy(padded_lbls_seqs)

        return padded_seqs, lens, lbls,file_names,padded_lbls_seqs


def main():
    return Dataset


if __name__ == "__main__":
    main()