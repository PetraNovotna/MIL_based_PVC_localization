clc;clear all;close all;

data_folder='../../Training_WFDB_filtered_2';

heatmap_names=subdir('../../res/*.npy');



heatmap_names={heatmap_names.name};

signal_names={};
for k = 1:length(heatmap_names)
    [filepath,name,ext] = fileparts(heatmap_names{k});
    
    tmp=[data_folder filesep replace(name,'_heatmap','') '.mat'];
    
    signal_names=[signal_names tmp];
    
    
    
end

super_kes_klikac(signal_names, heatmap_names)
