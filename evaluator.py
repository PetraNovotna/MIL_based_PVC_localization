import utils.load_fncs as lf
import glob
import numpy as np
from bayes_opt import BayesianOptimization
from scipy.signal import find_peaks
from utils.get_results import get_results


# hetmap_folder='../res_detection_gausian100'
hetmap_folder='../res_MIL_max'
# hetmap_folder='../res_MIL_mean'
# hetmap_folder='../res_MIL_maxzeros'
gt_folder = '../data_ke_clanku/output_labeled'


file_list_heatmap = glob.glob(hetmap_folder + "/*.npy")


lbls=[]
heatmaps=[]
for k,heatmap_name in enumerate(file_list_heatmap):

    gt_name = heatmap_name.replace(hetmap_folder,gt_folder).replace('_heatmap.npy','_position_labels.mat')
    
    
    lbl_PAC,lbl_PVC = lf.read_lbl_pos(gt_name)
    heatmap=np.load(heatmap_name)
    
   
    lbl = lbl_PAC
    heatmap = heatmap[[0],:]
    
    
    # lbl = lbl_PVC
    # heatmap = heatmap[[1],:]
    
    
    
    
    lbls.append(lbl)
    heatmaps.append(heatmap)
    

heat_max=-np.inf
heat_min=np.inf
for k,heatmap in enumerate(heatmaps):
    
    tmp=np.max(heatmap)
    if tmp>heat_max:
        heat_max=tmp
        
    tmp=np.min(heatmap)
    if tmp<heat_max:
        heat_min=tmp
    



def get_peaks(heatmaps,height,distance,prominence):
    detections=[]
    for k,heatmap in enumerate(heatmaps):
        
        peaks,properties=find_peaks(heatmap[0,:],height=height,distance=distance,prominence=prominence) 
        detections.append(peaks)
    
    return detections
    



def func(all_results=False,**params):
    detections = get_peaks(heatmaps,params['height'],params['distance'],params['prominence'])
    recall, precision, dice, acc = get_results(lbls,detections)
    
    if all_results:
        return recall, precision, dice, acc
    else :
        return dice
    


param_names=['height','distance','prominence']

bounds_lw=[heat_min,1,0]
bounds_up=[heat_max,1000,heat_max-heat_min]


pbounds=dict(zip(param_names, zip(bounds_lw,bounds_up)))    

optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)  

optimizer.maximize(init_points=200,n_iter=200)


print(optimizer.max)

params=optimizer.max['params']


recall, precision, dice, acc = func(all_results=True,**params)

print('recall ' + str(recall))
print('precision ' + str(precision))
print('dice ' + str(dice))
print('acc ' + str(acc))











