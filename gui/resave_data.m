clc;clear all;close all;


train_folder='../../data_filtered_train';
valid_folder='../../data_filtered_valid';

mkdir(train_folder)
mkdir(valid_folder)

data_folder='../../Training_WFDB_filtered_2';


heatmap_names=subdir('../../res/*.npy');
heatmap_names={heatmap_names.name};



data_names=subdir([data_folder '/*.mat']);
data_names={data_names.name};


for k = 1:length(data_names)
    [filepath,name_data,ext] = fileparts(data_names{k});
    
    has_heatmap=0;
    for kk = 1:length(heatmap_names)
        [filepath2,name_heatmap,ext2] = fileparts(heatmap_names{kk});
        
        if strcmp([name_data '_heatmap'],name_heatmap)
            has_heatmap=1;
        end
    end
    
    if has_heatmap
        tmp=data_names{k};
        copyfile(tmp,[valid_folder '/' name_data '.mat'])
        tmp=replace(data_names{k},'.mat','.hea');
        copyfile(tmp,[valid_folder '/' name_data '.hea'])
    else
        tmp=data_names{k};
        copyfile(tmp,[train_folder '/' name_data '.mat'])
        tmp=replace(data_names{k},'.mat','.hea');
        copyfile(tmp,[train_folder '/' name_data '.hea'])
    end
    
    
end