import os
from utils.losses import wce,mse
import numpy as np
import shutil

class Config:
    
    best_models_dir = "../models"

    model_save_dir = "../tmp"

    DATA_PATH = "../Training_WFDB"
    
    lbls_path='../output_klikace_all'


    is_mil=1
    
    mil_solution='max'
    
    gaussian_sigma=100
    
    
    # pato_names=['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
    # DATA_TMP_PATH = "../Training_WFDB_filtered"

    pato_names = ['Normal', 'PVC']
    DATA_TMP_PATH = "../Training_WFDB_filtered_2"
    
    
    
    
    if is_mil:
        res_dir='../res_MIL' + '_' + mil_solution
    else:
        res_dir='../res_detection_gausian' + str(gaussian_sigma)
        

    try:
        os.mkdir(DATA_TMP_PATH)
    except:
        pass
    
    
    try:
        shutil.rmtree(res_dir)
    except:
        pass
    
    
    
    try:
        os.mkdir(res_dir)
    except:
        pass
    
    try:
        os.mkdir(model_save_dir)
    except:
        pass

  



    train_batch_size = 32
    valid_batch_size = 32
    
    valid_num_workers = 3
    train_num_workers = 3


    
    LR_LIST=np.array([0.001,0.0001,0.00001])
    LR_CHANGES_LIST=[60,30,15]
    if is_mil:
        LOSS_FUNTIONS=[wce,wce,wce]
    else:
        LOSS_FUNTIONS=[mse,mse,mse]
    max_epochs=np.sum(LR_CHANGES_LIST)
    
    
    SPLIT_RATIO = [8, 2]

    model_note = 'test1'



    ## network setting
    levels = 6
    lvl1_size = 6
    input_size = 12
    output_size = 1
    convs_in_layer = 2
    init_conv = lvl1_size
    filter_size = 5



