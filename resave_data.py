# resave data
from shutil import copyfile, rmtree
import os
from config import Config
import load_fncs as lf




try:
    rmtree(Config.DATA_TMP_PATH)
except:
    pass

try:
    os.mkdir(Config.DATA_TMP_PATH)
except:
    pass


##get all file names
names=[]
for root, dirs, files in os.walk(Config.DATA_PATH):
    for name in files:
        if name.endswith(".mat"):
            name=name.replace('.mat','')
            names.append(name)



for k,file_name in enumerate(names):

    lbl = lf.read_lbl(Config.DATA_PATH, file_name)
    lbl = lbl[:-1]

    print(file_name)
    print(lbl)
    print(k)

    for pato_name in Config.pato_names:

      if pato_name==lbl:
        copyfile(Config.DATA_PATH + os.sep +file_name +'.mat',Config.DATA_TMP_PATH + os.sep +file_name +'.mat')
        copyfile(Config.DATA_PATH + os.sep +file_name + '.hea' ,Config.DATA_TMP_PATH + os.sep +file_name + '.hea')
        print("saved")

