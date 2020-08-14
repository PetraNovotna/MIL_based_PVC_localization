from torch.utils import data
import numpy as np
import torch
import config as Config
import load_fncs as lf
import os


class Dataset(data.Dataset):

    def __init__(self, list_of_ids, data_path, split):
        """Initialization"""
        self.path = data_path
        self.list_of_ids = list_of_ids
        self.split = split

        self.MEANS = np.load(Config.info_save_dir + os.sep + 'MEANS.npy')

        self.STDS = np.load(Config.info_save_dir + os.sep + 'STDS.npy')

        self.pato_names = Config.pato_names

    def __len__(self):
        """Return total number of data samples"""
        return len(self.list_of_ids)

    def __getitem__(self, idx):
        """Generate data sample"""
        # Select sample
        file_name = self.list_of_ids[idx]

        # Read data and get label
        X = lf.read_data(self.path, file_name)

        sig_len = X.shape[1]
        signal_num = X.shape[0]

        ##augmentation
        if self.split == 'train':
            ##random circshift
            if torch.rand(1).numpy()[0] > 0.3:
                shift = torch.randint(sig_len, (1, 1)).view(-1).numpy()

                X = np.roll(X, shift, axis=1)

            # ranzomly inserted zeros
            if torch.rand(1).numpy()[0] > 0.3:

                rand_pos = int(torch.randint(sig_len, (1, 1)).view(-1).numpy())
                if sig_len - rand_pos > Config.zeros_len:
                    X[:, rand_pos:rand_pos + Config.zeros_len] = 0
                else:
                    X[:, rand_pos:] = 0

            ## random stretch -
            if torch.rand(1).numpy()[0] > 0.3:

                max_resize_change = 0.2
                relative_change = 1 + torch.rand(1).numpy()[0] * 2 * max_resize_change - max_resize_change
                new_len = int(relative_change * sig_len)

                Y = np.zeros((signal_num, new_len))
                for k in range(signal_num):
                    Y[k, :] = np.interp(np.linspace(0, sig_len - 1, new_len), np.linspace(0, sig_len - 1, sig_len),
                                        X[k, :])
                X = Y

            ## random multiplication of each lead by a number
            if torch.rand(1).numpy()[0] > 0.3:

                max_mult_change = 0.2

                for k in range(signal_num):
                    mult_change = 1 + torch.rand(1).numpy()[0] * 2 * max_mult_change - max_mult_change
                    X[k, :] = X[k, :] * mult_change

        ## normalization
        X = (X - self.MEANS.reshape(-1, 1)) / self.STDS.reshape(-1, 1)

        ## laod label
        lbl = lf.read_lbl(self.path, file_name)
        # print(lbl)
        lbl = lbl.split(',')

        ## create more hot encoding
        y = np.zeros((len(self.pato_names), 1)).astype(np.float32)
        for kk, p in enumerate(self.pato_names):
            for lbl_i in lbl:
                if lbl_i.find(p) > -1:
                    y[kk] = 1

        y = y[1, 0].reshape(1, 1)

        # print(y)

        return X, y

    def collate_fn(data):
        ## this take list of samples and put them into batch

        ##pad with zeros
        pad_val = 0

        ## get list of singals and its lengths
        seqs, lbls = zip(*data)

        lens = [seq.shape[1] for seq in seqs]

        ## pad shorter signals with zeros to make them same length
        padded_seqs = pad_val * np.ones((len(seqs), seqs[0].shape[0], np.max(lens))).astype(np.float32)
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :, :end] = seq

        ## stack and reahape signal lengts to 10 vector
        lbls = np.stack(lbls, axis=0)
        lbls = lbls.reshape(lbls.shape[0:2])
        lens = np.array(lens).astype(np.float32)

        ## numpy -> torch tensor
        padded_seqs = torch.from_numpy(padded_seqs)
        lbls = torch.from_numpy(lbls)
        lens = torch.from_numpy(lens)

        return padded_seqs, lens, lbls


def main():
    return Dataset


if __name__ == "__main__":
    main()