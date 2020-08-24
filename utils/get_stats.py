from torch.utils import data 
import numpy as np
from config import Config
from dataloader import Dataset
from copy import deepcopy

def get_stats(file_names_list):

    validation_generator = Dataset(file_names_list, 'valid')
    validation_generator = data.DataLoader(validation_generator, batch_size=Config.valid_batch_size,
                                           num_workers=Config.valid_num_workers, shuffle=False, drop_last=False,
                                           collate_fn=Dataset.collate_fn)
    
    
    one_hots=[]
    lenss=[]
    for it,(pad_seqs, lens, lbls, file_names,detection) in enumerate(validation_generator):
        print(it)
        
        one_hots.append(deepcopy(lbls.detach().cpu().numpy()))
        lenss.append(deepcopy(lens.detach().cpu().numpy()))
        del lbls
        del lens
        del pad_seqs
    
        
    one_hots=np.concatenate(one_hots,axis=0)
    lenss=np.concatenate(lenss,axis=0)
    lbl_counts=np.sum(one_hots,0)
    
    return lbl_counts,lenss