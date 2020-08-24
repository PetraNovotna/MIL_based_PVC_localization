import numpy as np
import time
import os


class AdjustLearningRateAndLoss():
    def __init__(self,optimizer,learning_rates_list,lr_changes_list,loss_functions):
        self.optimizer=optimizer
        self.learning_rates_list=learning_rates_list
        self.lr_changes_list=lr_changes_list
        self.loss_functions=loss_functions
        
        self.actual_loss=self.loss_functions[0]
        self.actual_lr=self.learning_rates_list[0]
        self.lr_changes_cumulative=np.cumsum([0] +self.lr_changes_list)
        self.epoch=0
        
    def step(self):
        self.epoch=self.epoch+1

        try:
            with open('put_here_to_aply/lr_change.txt', 'r') as f:
                x = f.readlines()
                lr=float(x[0])
            time.sleep(1)
            os.remove('put_here_to_aply/lr_change.txt')
            
            print('lr was set to: ' + str(x))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        except:
            pass
        
        
        for ind in range(len(self.lr_changes_cumulative)-1):
            value1=self.lr_changes_cumulative[ind]
            value2=self.lr_changes_cumulative[ind+1]
            if self.epoch>=value1 and self.epoch<value2:
                self.actual_loss=self.loss_functions[ind]
                self.actual_lr=self.learning_rates_list[ind]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] =self.actual_lr