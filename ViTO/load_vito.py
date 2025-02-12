import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from unet import UNET
import sys
import time


data = sio.loadmat('../../../New_data/Portion1_2250/data1.mat')
strain = data['strain']
stress_train = data['stress_train']
stress_test = data['stress_test']
stress_weight_train = data['stress_weight_train']
stress_weight_test = data['stress_weight_test']
phase_train = data['phase_train']
phase_test = data['phase_test']
angle_train = data['angle_train']
angle_test = data['angle_test']
num_pts = data['num_pts']
num_cs = data['num_cs']
strain = strain.squeeze()
num_cs = num_cs[0,0]
num_pts = num_pts[0,0]

TS_train = phase_train # (2250, 51, 51, 7, 3)

F_train = stress_train # (2250, 3, 2, 51)

TS_test = phase_test   # (250, 51, 51, 7, 3)

F_test = stress_test   # (250, 3, 2, 51)






TS_train = np.transpose(TS_train,axes=[0,3,4,1,2])
TS_test = np.transpose(TS_test,axes=[0,3,4,1,2])

TS_train = np.reshape(TS_train,(TS_train.shape[0],21,TS_train.shape[3],TS_train.shape[4]))
TS_test = np.reshape(TS_test,(TS_test.shape[0],21,TS_test.shape[3],TS_test.shape[4]))




BS_TRAIN = 40
BS_TEST = 10

sh = TS_train.shape
shuffler = np.random.permutation(sh[0])

#TS_train = TS[shuffler,:,:,:]
#F_train = F[shuffler,:,:,:]

TS_train = TS_train[:,:,0:48,0:48]
TS_test = TS_test[:,:,0:48,0:48]

par = {
       'inp_ch'     : TS_train.shape[1],
       'out_ch'     : F_train.shape[1],
       'n_channels' : 8,
       'k'          : 3,
       'inp_shift'  : np.min(TS_train),
       'inp_scale'  : np.max(TS_train)-np.min(TS_train),
       'out_shift'  : np.min(F_train),
       'out_scale'  : np.max(F_train) - np.min(F_train)
       }




TS_train = torch.tensor(TS_train).to('cuda:0', dtype=torch.float)
TS_test = torch.tensor(TS_test).to('cuda:0', dtype=torch.float)

F_train = torch.tensor(F_train).to('cuda:0', dtype=torch.float)
F_test = torch.tensor(F_test).to('cuda:0', dtype=torch.float)

#Y_train_weight = torch.tensor(Y_train_weight).to('cuda:0', dtype=torch.float)
Y_test_weight = torch.tensor(stress_weight_test).to('cuda:0', dtype=torch.float)

strain = torch.tensor(strain).to('cuda:0', dtype=torch.float)


#model = Vito(par).cuda()
my_scripted_model = UNET(par).cuda()#torch.jit.script(model)
my_scripted_model.load_state_dict(torch.load("best_param.pt"))

D = my_scripted_model(TS_test, strain)
D = D.cpu().data.numpy()

np.save('ViTO_pred.npy', D, allow_pickle=True)


print('################### done #################################')

                                                                             
































