import os


class Config:
    best_models_dir = "../models"

    model_save_dir = "../tmp"

    DATA_PATH = "../Training_WFDB"

    DATA_TMP_PATH = "../Training_WFDB_filtered_2"
    # DATA_TMP_PATH= "/content/drive/My Drive/CinC2020_semisupervised_SVES_KES/data/Training_WFDB"
    # DATA_TMP_PATH= "/content/drive/My Drive/CinC2020_semisupervised_SVES_KES/data/Training_WFDB_filtered_PAC_normal"

    try:
        os.mkdir(DATA_TMP_PATH)
    except:
        pass

    info_save_dir = "../tmp_info"

    # PVC  vs  ostatní data bez LBBB, RBBB, PAC

    # PVC  vs  normal

    # pato_names=['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']

    pato_names = ['Normal', 'PVC']

    train_batch_size = 32
    train_num_workers = 4
    valid_batch_size = 32
    valid_num_workers = 4

    max_epochs = 107
    step_size = 35
    gamma = 0.1
    init_lr = 0.01

    split_ratio = [8, 2]

    model_note = 'test1'

    ## network setting
    levels = 6
    lvl1_size = 4
    input_size = 12
    output_size = 1
    convs_in_layer = 2
    init_conv = 4
    filter_size = 5

    zeros_len = 400

    ploting = False

