import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from unet import UNET 
import sys
import time


seed = 23
np.random.seed(seed)
torch.manual_seed(seed)

data = sio.loadmat('../New_data/Portion1_2250/data1.mat')
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

TS_train = TS_train[:,:,0:48,0:48]
TS_test = TS_test[:,:,0:48,0:48]



par = {
       'inp_ch'     : TS_train.shape[1],
       'out_ch'     : 6,#F_train.shape[1],
       'n_channels' : 8,
       'k'          : 3,
       'inp_shift'  : np.min(TS_train),
       'inp_scale'  : np.max(TS_train)-np.min(TS_train),
       'out_shift'  : np.min(F_train),
       'out_scale'  : np.max(F_train) - np.min(F_train)
       }

np.save('True/F_true_train.npy', F_train, allow_pickle=True)
np.save('Test/F_true_test.npy', F_test, allow_pickle=True)

np.save('True/TS_true_train.npy', TS_train, allow_pickle=True)
np.save('Test/TS_true_test.npy', TS_test, allow_pickle=True)

print('Shape of input: ', np.shape(TS_train))
print('Shape of output: ', np.shape(F_train))

TS_train = torch.tensor(TS_train).to('cuda:0', dtype=torch.float)
TS_test = torch.tensor(TS_test).to('cuda:0', dtype=torch.float)

F_train = torch.tensor(F_train).to('cuda:0', dtype=torch.float)
F_test = torch.tensor(F_test).to('cuda:0', dtype=torch.float)

Y_train_weight = torch.tensor(stress_weight_train).to('cuda:0', dtype=torch.float)
Y_test_weight = torch.tensor(stress_weight_test).to('cuda:0', dtype=torch.float)

strain = torch.tensor(strain).to('cuda:0', dtype=torch.float)

#model = Vito(par).cuda()
my_scripted_model = UNET(par).cuda()#torch.jit.script(model)

loss_function = my_scripted_model.loss


optimizer = torch.optim.Adam(my_scripted_model.parameters(), lr = 5e-4, weight_decay = 1e-7)

epochs = 150000

train_loss_ls = []
test_loss_ls = []


lowest_loss = 1000
best_model_id = 0

begin_time = time.time()
for epoch in range(epochs):
    ep_time0 = time.time()
    D_pred  = my_scripted_model(TS_train, strain)
   
    # Calculating the loss function
    loss = loss_function(F_train, D_pred, Y_train_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ep_time_1 =  time.time()
    if epoch%100==0:
        ep_time100 = time.time()
        D_test_pred = my_scripted_model(TS_test, strain)

        test_loss = loss_function(F_test, D_test_pred, Y_test_weight)

        l_train = loss.cpu().data.numpy()
        l_test = test_loss.cpu().data.numpy()

        train_loss_ls.append(l_train)
        test_loss_ls.append( l_test )

        if l_test < lowest_loss:
            best_model_id = epoch
            lowest_loss = l_test
            
            torch.save(my_scripted_model.state_dict(), 'Params/Vito_'+str(epoch)+'.pt')
   
        print("epoch: "+str(epoch)+", train loss: "+"{:.3e}".format(l_train)+", val loss: "+"{:.3e}".format(l_test)+", best model: "+str(best_model_id)+", lowest error: "+"{:.3e}".format(lowest_loss)+", elapsed time: "+"{:.3e}".format(time.time()-begin_time)+"s", "per 100 ep time: "+"{:.3e}".format(ep_time100-ep_time_1)+"s" )
print('################### done #################################')

                                                                             
































