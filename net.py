import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt


class myConv(nn.Module):
    def __init__(self, in_size, out_size, filter_size=3, stride=1, pad=None, do_batch=1, dov=0):
        super().__init__()

        pad = int((filter_size - 1) / 2)

        self.do_batch = do_batch
        self.dov = dov
        self.conv = nn.Conv1d(in_size, out_size, filter_size, stride, pad)
        self.bn = nn.BatchNorm1d(out_size, momentum=0.1)

        if self.dov > 0:
            self.do = nn.Dropout(dov)

    def forward(self, inputs):

        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)

        outputs = F.relu(outputs)

        if self.dov > 0:
            outputs = self.do(outputs)

        return outputs


class Net_addition_grow(nn.Module):

    def __init__(self, levels=7, lvl1_size=4, input_size=12, output_size=9, convs_in_layer=3, init_conv=4,
                 filter_size=13, mil_solution='max'):
        super().__init__()
        self.mil_solution=mil_solution
        self.levels = levels
        self.lvl1_size = lvl1_size
        self.input_size = input_size
        self.output_size = output_size
        self.convs_in_layer = convs_in_layer
        self.filter_size = filter_size

        self.t = 0.5 * np.ones(output_size)

        self.init_conv = myConv(input_size, init_conv, filter_size=filter_size)

        self.layers = nn.ModuleList()
        for lvl_num in range(self.levels):

            if lvl_num == 0:
                self.layers.append(myConv(init_conv, int(lvl1_size * (lvl_num + 1)), filter_size=filter_size))
            else:
                self.layers.append(myConv(int(lvl1_size * (lvl_num)) + int(lvl1_size * (lvl_num)) + init_conv,
                                          int(lvl1_size * (lvl_num + 1)), filter_size=filter_size))

            for conv_num_in_lvl in range(self.convs_in_layer - 1):
                self.layers.append(
                    myConv(int(lvl1_size * (lvl_num + 1)), int(lvl1_size * (lvl_num + 1)), filter_size=filter_size))

        self.conv_final = nn.Conv1d(int(lvl1_size * (self.levels)) + int(lvl1_size * (self.levels)) + init_conv, 1, 3,
                                    1, 1)

        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x, lens,detection):

        ## make signal len be divisible by 2**number of levels
        ## replace rest by zeros
        for signal_num in range(list(x.size())[0]):
            k = int(np.floor(lens[signal_num].cpu().numpy() / (2 ** (self.levels - 1))) * (2 ** (self.levels - 1)))

            x[signal_num, :, k:] = 0

        ## pad with more zeros  -  add as many zeros as convolution of all layers can proppagete numbers
        n = (self.filter_size - 1) / 2
        padded_length = n
        for p in range(self.levels):
            for c in range(self.convs_in_layer):
                padded_length = padded_length + 2 ** p * n
        padded_length = padded_length + 2 ** p * n + 256  # 256 for sure

        shape = list(x.size())
        xx = torch.zeros((shape[0], shape[1], int(padded_length)), dtype=x.dtype)
        cuda_check = x.is_cuda
        if cuda_check:
            cuda_device = x.get_device()
            device = torch.device('cuda:' + str(cuda_device))
            xx = xx.to(device)

        x = torch.cat((x, xx), 2)  ### add zeros to signal
        
        
        
        shape = list(detection.size())
        xx = torch.zeros((shape[0], shape[1], int(padded_length)), dtype=x.dtype)
        cuda_check = detection.is_cuda
        if cuda_check:
            cuda_device = detection.get_device()
            device = torch.device('cuda:' + str(cuda_device))
            xx = xx.to(device)
        detection=torch.cat((detection, xx), 2)
        detection=F.avg_pool1d(detection, 2 ** self.levels, 2 ** self.levels)




        x.requires_grad = True

        x = self.init_conv(x)

        x0 = x

        ## aply all convolutions
        layer_num = -1
        for lvl_num in range(self.levels):

            for conv_num_in_lvl in range(self.convs_in_layer):
                layer_num += 1
                if conv_num_in_lvl == 1:
                    y = x

                x = self.layers[layer_num](x)

            ## skip conection to previous layer and to the input
            x = torch.cat((F.avg_pool1d(x0, 2 ** lvl_num, 2 ** lvl_num), x, y), 1)

            x = F.max_pool1d(x, 2, 2)

        x = self.conv_final(x)

        # x=F.relu(x)

        heatmap = x

        ### replace padded parts of signals by -inf => it will be not used in poolig
        for signal_num in range(list(x.size())[0]):
            k = int(np.floor(lens[signal_num].cpu().numpy() / (2 ** (self.levels - 1))))

            x[signal_num, :, k:] = 0
        

        # set_trace()

        help_mult = list(x.size())[2]

        if self.mil_solution=='max':
            x = F.adaptive_max_pool1d(x, 1)
        elif self.mil_solution=='mean':
            x = F.adaptive_avg_pool1d(x, 1)
        else:
            print('error mil solution')
        

        help_lens = lens.view(list(x.size()))

        x = (x * help_mult) / help_lens

        x = x.view(list(x.size())[:2])
        score = x

        x = torch.sigmoid(x)
        # x= (torch.sigmoid(x) - 0.5) * 2

        return x, heatmap, score,detection

    def save_log(self, log):
        self.log = log

    def save_config(self, config):
        self.config = config

    def plot_training(self):

        plt.plot(self.log.trainig_loss_log, 'b')
        plt.plot(self.log.valid_loss_log, 'r')
        plt.title('loss')
        plt.show()

        plt.plot(self.log.trainig_beta_log, 'b')
        plt.plot(self.log.valid_beta_log, 'g')
        plt.title('geometric mean')
        plt.show()

    def save_plot_training(self, name):

        plt.plot(self.log.trainig_loss_log, 'b')
        plt.plot(self.log.valid_loss_log, 'r')
        plt.title('loss')
        plt.savefig(name + '_loss.png')

        plt.plot(self.log.trainig_beta_log, 'b')
        plt.plot(self.log.valid_beta_log, 'g')
        plt.title('geometric mean')
        plt.savefig(name + '_geometric_mean.png')