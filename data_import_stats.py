
# data import and stats
import os
import numpy as np
from config import Config
import load_fncs as lf

try:
    os.mkdir(Config.info_save_dir)
except:
    pass

##get all file names
names = []
for root, dirs, files in os.walk(Config.DATA_TMP_PATH):
    for name in files:
        if name.endswith(".mat"):
            name = name.replace('.mat', '')
            names.append(name)

## measure signal statistics for normalization
labels = []
means = []
stds = []
lens = []
for k, file_name in enumerate(names):
    X = lf.read_data(Config.DATA_TMP_PATH, file_name)

    means.append(np.mean(X, axis=1))
    stds.append(np.std(X, axis=1))
    lens.append(X.shape[1])

    lbl = lf.read_lbl(Config.DATA_TMP_PATH, file_name)

    labels.append(lbl)

MEANS = np.mean(np.stack(means, axis=1), axis=1)
STDS = np.mean(np.stack(stds, axis=1), axis=1)

## create more-hot-encoding to measure count of each pathology in data
more_hot_lbls = []
for k, lbl in enumerate(labels):

    res = np.zeros(len(Config.pato_names))

    lbl = lbl.split(',')

    for kk, p in enumerate(Config.pato_names):
        for lbl_i in lbl:
            if lbl_i.find(p) > -1:
                res[kk] = 1

    more_hot_lbls.append(res > 0)

tmp = np.stack(more_hot_lbls, axis=1)

lbl_counts = np.sum(tmp, axis=1)

num_of_sigs = len(lens)

print(MEANS)

print(STDS)

print(lbl_counts)

print(len(lens))

## save statistics
np.save(Config.info_save_dir + os.sep + 'MEANS.npy', np.array(MEANS))
np.save(Config.info_save_dir + os.sep + 'STDS.npy', np.array(STDS))
np.save(Config.info_save_dir + os.sep + 'lbl_counts.npy', np.array(lbl_counts))
np.save(Config.info_save_dir + os.sep + 'lens.npy', np.array(lens))

np.random.seed(666)

split_ratio_ind = int(np.floor(Config.split_ratio[0] / (Config.split_ratio[0] + Config.split_ratio[1]) * len(names)))
perm = np.random.permutation(len(names))
train_ind = perm[:split_ratio_ind]
valid_ind = perm[split_ratio_ind:]
split = {'train': [names[i] for i in train_ind], 'valid': [names[i] for i in valid_ind]}

np.save(Config.info_save_dir + os.sep + 'split.npy', split)

np.save(Config.info_save_dir + os.sep + 'num_of_sigs.npy', num_of_sigs)