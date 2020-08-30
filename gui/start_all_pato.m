clc;clear all;close all force;


data_dir='../../Training_WFDB';

net_prediction_dir='../../net_prediction_for_gui';

output_dir='../../output_labeled';

backup_dir='../../backup';


mkdir(output_dir)
mkdir(backup_dir)

PVC_PAC_AF_labeler(data_dir, net_prediction_dir,output_dir,backup_dir)