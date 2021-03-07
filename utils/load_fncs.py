# load fcns
import os
import scipy.io as io



def read_data(file_name):
    data_dict = io.loadmat(file_name)
    return data_dict["val"]

def read_lbl_orig(file_name):
    name=file_name.replace('.mat','.hea')

    # Read line 15 in header file and parse string with labels
    with open(name, "r") as file:
        for line_idx, line in enumerate(file):
            if line_idx == 15:
                line=line.replace('#Dx: ','')
                lbl=line
                break
    file.close()
    lbl = lbl[:-1]
    lbl = lbl.split(',')
    
    return lbl


def read_lbl_pos(file_name):
    
    data_dict = io.loadmat(file_name)
    
    lbl_PAC,lbl_PVC = data_dict["result_positions_PAC"].flatten(),data_dict["result_positions_PVC"].flatten()
    
    return lbl_PAC,lbl_PVC
    
    