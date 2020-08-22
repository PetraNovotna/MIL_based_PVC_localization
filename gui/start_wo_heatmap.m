clc;clear all;close all force;




data_dir='../../data_filtered_train';

heatmap_dir='';

output_dir='../../output_klikace_train';
backup_dir='../../zalohy_train';

mkdir(output_dir)
mkdir(backup_dir)


super_kes_klikac(data_dir, heatmap_dir,output_dir,backup_dir)

