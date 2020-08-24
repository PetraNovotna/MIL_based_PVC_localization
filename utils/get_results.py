from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import numpy as np

def get_results(lbls,detetions):
    
    
    dist_t=250
    
    TP=0
    FP=0
    FN=0
    
    for k,(lbl,detetion) in enumerate(zip(lbls,detetions)):
    
        
        lbl = lbl.reshape(-1, 1)
        
        detetion = detetion.reshape(-1, 1)
        
        if len(lbl)==0:
            FP = FP + len(detetion)
            continue
            
        if len(detetion)==0:
            FN = FN + len(lbl)
            continue
            
        D=pairwise_distances(lbl,detetion)
        
        D[D>dist_t]=10**5
        
        
        row_ind, col_ind = linear_sum_assignment(D)
        
        d=D[row_ind,col_ind]
        
        row_ind, col_ind = row_ind[d<=dist_t],col_ind[d<=dist_t]
        
        d=D[row_ind,col_ind]
        
        
        num_lbl=len(lbl)
        num_det=len(detetion)
        num_pairs=len(row_ind)
        
        FP = FP + (num_det-num_pairs)
        FN = FN + (num_lbl-num_pairs)
        TP = TP + num_pairs
        
        
    if (TP + FN) ==0:
        recall=0
    else:
        recall=TP / (TP + FN)
       
    if (TP + FP) ==0:
        precision=0
    else:
        precision=TP / (TP + FP)
        
    dice= 2* TP / (2*TP + FP + FN)
    acc=np.mean(np.array([len(x)>0 for x in detetions])==np.array([len(x)>0 for x in lbls]))
    
    
    return recall, precision, dice, acc
    
    