clc;clear all;close all force;




data_dir='../../data_filtered_valid';

heatmap_dir='../../res';

output_dir='../../output_klikace_valid';

backup_dir='../../zalohy_valid';

mkdir(output_dir)
mkdir(backup_dir)

super_kes_klikac(data_dir, heatmap_dir,output_dir,backup_dir)


