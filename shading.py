import os
import numpy as np


import torch
from torch import optim
from torch.utils import data
import load_fncs as lf
from config import Config

split = np.load(Config.info_save_dir + os.sep + 'split.npy', allow_pickle=True).item()

model_name = '/content/drive/My Drive/CinC2020_semisupervised_SVES_KES/models/KES_model_trained.pkl'
# model_name = '/content/drive/My Drive/CinC2020_semisupervised_SVES_KES/models/SVES_model_trained.pkl'

# newdir_path = '/content/drive/My Drive/CinC2020_semisupervised_SVES_KES/res_fig'
newdir_path = '/content/drive/My Drive/CinC2020_semisupervised_SVES_KES/res_fig_2'

try:
    os.mkdir(newdir_path)
except:
    pass

# device = torch.device("cuda:"+str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

model = torch.load(model_name, map_location=device)

### set model to eval mode and send it to graphic card
model = model.eval().to(device)
obr = ["A0310", "A0452"]

for name in split['valid']:
#for name in obr:

    # Read data and get label
    data = lf.read_data(Config.DATA_TMP_PATH, name)

    ## laod label
    lbl = lf.read_lbl(Config.DATA_TMP_PATH, name)
    # print(lbl)
    lbl = lbl.split(',')

    ## create more hot encoding
    y = np.zeros((len(Config.pato_names), 1)).astype(np.float32)
    for kk, p in enumerate(Config.pato_names):
        for lbl_i in lbl:
            if lbl_i.find(p) > -1:
                y[kk] = 1

    y = y[1, 0].reshape(1, 1)

    ## load data statisctics for normalization
    MEANS = np.load(Config.info_save_dir + os.sep + 'MEANS.npy')
    STDS = np.load(Config.info_save_dir + os.sep + 'STDS.npy')
    pato_names = model.config.pato_names
    lens_all = np.load(Config.info_save_dir + os.sep + 'lens.npy')
    batch = model.config.train_batch_size

    data0 = (data - MEANS.reshape(-1, 1)) / STDS.reshape(-1, 1).copy()

    lens_sample = np.random.choice(lens_all, batch, replace=False)
    max_len = np.max(lens_sample)
    data_new = np.zeros((data0.shape[0], max(max_len, data0.shape[1])))
    data_new[:, :data0.shape[1]] = data0
    data_np = data_new.copy()

    ## model require signal len for removal of padded part before max pooling
    lens = data0.shape[1]
    lens_np = data0.shape[1]
    f = len(np.arange(0, lens, 50))

    heatmap_res = np.zeros(f)
    heatmap_score = np.zeros(f)

    win_len = Config.zeros_len
    for ind, pos in enumerate(range(0, lens, 50)):
        win_half = int(Config.zeros_len / 2)

        pos_vec = np.arange(pos - win_half, pos + win_half)
        pos_vec = pos_vec[pos_vec >= 0]
        pos_vec = pos_vec[pos_vec < lens_np]

        data_np_tmp = data_np.copy()
        data_np_tmp[:, pos_vec] = 0

        ## numpy array => tensor
        lens = torch.from_numpy(np.array(lens_np).astype(np.float32)).view(1)
        data = torch.from_numpy(
            np.reshape(data_np_tmp.astype(np.float32), (1, data_np_tmp.shape[0], data_np_tmp.shape[1])))

        lens = lens.to(device)
        data = data.to(device)

        res, heatmap, score = model(data, lens)

        res = res.detach().cpu().numpy()[0, 0]
        score = score.detach().cpu().numpy()[0, 0]

        heatmap_res[ind] = res
        heatmap_score[ind] = score

    heatmap_res = np.interp(np.linspace(0, len(heatmap_res) - 1, lens_np),
                            np.linspace(0, len(heatmap_res) - 1, len(heatmap_res)), heatmap_res)
    heatmap_score = np.interp(np.linspace(0, len(heatmap_score) - 1, lens_np),
                              np.linspace(0, len(heatmap_score) - 1, len(heatmap_score)), heatmap_score)

    lens = torch.from_numpy(np.array(lens_np).astype(np.float32)).view(1)
    data = torch.from_numpy(np.reshape(data_np.astype(np.float32), (1, data_np.shape[0], data_np.shape[1])))

    lens = lens.to(device)
    data = data.to(device)

    res, heatmap, score = model(data, lens)

    heatmap = heatmap.detach().cpu().numpy()[0, 0, :]
    data = data.detach().cpu().numpy()[0, 1, :]

    len_short = int(lens.detach().cpu().numpy()[0])
    len_ratio = len_short / len(data)
    heatmap_end = int(round(len(heatmap) * len_ratio))
    heatmap = heatmap[0:heatmap_end]

    heatmap = np.interp(np.linspace(0, len(heatmap) - 1, lens_np), np.linspace(0, len(heatmap) - 1, len(heatmap)),
                        heatmap)

    print("-------------------------------------")
    # plt.figure(figsize = (15,7))
    # plt.plot(heatmap_res,'b')
    # plt.title(str(res.detach().cpu().numpy()[0,0]))
    # plt.show()

    x = np.linspace(0, (len_short - 1) / 500, len_short)

    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 22})
    plt.subplots_adjust(hspace=0.07)

    plt.subplot(2, 1, 1)
    plt.plot(x, heatmap / 1000, color=(0.8500, 0.3250, 0.0980))
    plt.axhline(y=0, color='k')
    # plt.title(str(res.detach().cpu().numpy()[0,0]) + ', ' + str(y[0,0]))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.ylabel("Position likehood")

    # plt.subplot(3,1,2)
    # plt.plot(-heatmap_score,'r')

    plt.subplot(2, 1, 2)
    plt.plot(x, data[0:len_short] * STDS[1] / 1000, color=(0, 0.4470, 0.7410))
    plt.axhline(y=0, color='k')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")

    # plt.show()

    # data_short = data[0:len_short]

    # heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    # data_short_norm = (data_short - np.min(data_short)) / (np.max(data_short) - np.min(data_short))

    # plt.figure(figsize = (15,5))

    # plt.plot(heatmap_norm,'g')
    # plt.title(str(res.detach().cpu().numpy()[0,0]) + ', ' + str(y[0,0]))
    # plt.plot(data_short_norm,'k')

    # plt.show()

    plt.savefig(newdir_path + os.sep + name + '_2.svg', dpi=300)

    np.save(newdir_path + os.sep + 'heatmap.npy', heatmap)
    np.save(newdir_path + os.sep + 'original_signal.npy', data)

    ## predict with model

