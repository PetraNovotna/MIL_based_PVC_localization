clc;clear all;close all;


input_dir='../../output_klikace_valid_all';



output_dir='../../output_labeled';




files=subdir(input_dir);
files={files(:).name};

for k=1:length(files)
   file =  files{k};
   
   
   [filepath,name,ext] = fileparts(file);
   
   
   name_save = [output_dir '/' replace(name,'_naklikane','_position_labels.mat')];
   
   tmp=load(file);
   
   result_positions_PVC=tmp.result_positionss;
   result_positions_PAC=[];
   result_positions_AF=[];
   
   save(name_save,'result_positions_PVC','result_positions_PAC','result_positions_AF')
   
   
   
end