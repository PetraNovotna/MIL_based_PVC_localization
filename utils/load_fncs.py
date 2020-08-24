# load fcns
import os
import scipy.io as io



def read_data(file_name):
    data_dict = io.loadmat(file_name)
    return data_dict["val"]

# def read_lbl(file_name):
#     name=file_name.replace('.mat','.hea')

#     # Read line 15 in header file and parse string with labels
#     with open(name, "r") as file:
#         for line_idx, line in enumerate(file):
#             if line_idx == 15:
#                 line=line.replace('#Dx: ','')
#                 lbl=line
#                 break
#     file.close()
#     return lbl


def read_lbl(file_name):
    data_dict = io.loadmat(file_name)
    return data_dict["result_positionss"]
    
    