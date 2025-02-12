import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from pca_don import BranchNet, PCAFixedTrunkNet 
import time
import sys
import scipy.io as sio
import sys
import time


seed = 23
np.random.seed(seed)
torch.manual_seed(seed)

data = sio.loadmat('../../New_data/Portion1_2250/data1.mat')
strain = data['strain']
stress_train = data['stress_train']
stress_test = data['stress_test']
Y_train_weight = data['stress_weight_train']
Y_test_weight = data['stress_weight_test']
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

# Configuration and example tensors
num_channels = TS_train.shape[1]  # Number of channels in input
nx, ny = TS_train.shape[2], TS_train.shape[3]  # Spatial dimensions of the input
num_basis = 62 # Number of PCA basis components
# Dimensionality of the output space

#reshaped_data = F_train.reshape(F_train.shape[0] * F_train.shape[1], F_train.shape[2] * F_train.shape[3])

#print(reshaped_data.shape)

# Example PCA basis and mean function tensors
#pca_basis = torch.randn(num_basis, output_dim)  # Shape: [num_basis, output_dim]

data_01 = F_train[:,0,:,:]
data_02 = F_train[:,1,:,:]
data_03 = F_train[:,2,:,:]

r_data_01 = data_01.reshape(data_01.shape[0],data_01.shape[1]*data_01.shape[2])
r_data_02 = data_02.reshape(data_02.shape[0],data_02.shape[1]*data_02.shape[2])
r_data_03 = data_03.reshape(data_03.shape[0],data_03.shape[1]*data_03.shape[2])


N = 62
pca = PCA(n_components=N)
pca.fit(r_data_01)

# The pca.components_ attribute holds the PCA basis (principal components)
pca_basis_01 = pca.components_  # Shape: [N, n_features]

np.save('pca_basis_01.npy',pca_basis_01, allow_pickle=True)
print('trunk net shape:', pca_basis_01.T.shape)
#sys.exit()

pca = PCA(n_components=N)
pca.fit(r_data_02)

# The pca.components_ attribute holds the PCA basis (principal components)
pca_basis_02 = pca.components_  # 
# Optionally, convert pca_basis to a PyTorch tensor if you're using PyTorch
np.save('pca_basis_02.npy',pca_basis_02, allow_pickle=True)

pca = PCA(n_components=N)
pca.fit(r_data_03)

# The pca.components_ attribute holds the PCA basis (principal components)
pca_basis_03 = pca.components_  # 
# Optionally, convert pca_basis to a PyTorch tensor if you're using PyTorch
np.save('pca_basis_03.npy',pca_basis_03, allow_pickle=True)


mean_function_01 =  np.mean(r_data_01, axis=0) # Shape: [1, output_dim]
mean_function_02 =  np.mean(r_data_02, axis=0) # Shape: [1, output_dim]
mean_function_03 =  np.mean(r_data_03, axis=0) # Shape: [1, output_dim]



TS_train = torch.tensor(TS_train).to('cuda:0', dtype=torch.float)
TS_test = torch.tensor(TS_test).to('cuda:0', dtype=torch.float)

F_train = torch.tensor(F_train).to('cuda:0', dtype=torch.float)
F_test = torch.tensor(F_test).to('cuda:0', dtype=torch.float)

strain = torch.tensor(strain).to('cuda:0', dtype=torch.float)

Y_train_weight = torch.tensor(Y_train_weight).to('cuda:0', dtype=torch.float)
Y_test_weight = torch.tensor(Y_test_weight).to('cuda:0', dtype=torch.float)

pca_basis_tensor_01 = torch.tensor(pca_basis_01, dtype=torch.float32)
mean_function_tensor_01 = torch.tensor(mean_function_01, dtype=torch.float32)

pca_basis_tensor_02 = torch.tensor(pca_basis_02, dtype=torch.float32)
mean_function_tensor_02 = torch.tensor(mean_function_02, dtype=torch.float32)

pca_basis_tensor_03 = torch.tensor(pca_basis_03, dtype=torch.float32)
mean_function_tensor_03 = torch.tensor(mean_function_03, dtype=torch.float32)

# Initialize the branch network and the fixed trunk net model

branch_net = BranchNet(num_channels, nx, ny, num_basis).cuda()
#branch_net_02 = BranchNet(num_channels, nx, ny, num_basis).cuda()
#branch_net_03 = BranchNet(num_channels, nx, ny, num_basis).cuda()

import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model = PCAFixedTrunkNet(par, branch_net, pca_basis_tensor_01, mean_function_tensor_01, pca_basis_tensor_02, mean_function_tensor_02, pca_basis_tensor_03, mean_function_tensor_03).cuda()
model.apply(init_weights)
loss_function = model.loss

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-7)

epochs = 150000

train_loss_ls = []
test_loss_ls = []


lowest_loss = 1000
best_model_id = 0


# Assuming TS_train, train_labels, TS_val, and val_labels are already defined and are tensors
train_dataset = TensorDataset(TS_train, F_train)  # Wrap training data in a TensorDataset
val_dataset = TensorDataset(TS_test, F_test)       # Wrap validation data in a TensorDataset

batch_size = TS_train.shape[0]  # Define your batch size here

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader for training
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)     # Create DataLoader for validation

begin_time = time.time()

for epoch in range(epochs):
    ep_time_0 = time.time()

    # Training step
    model.train()  # Set the model to training mode
    for batch_data, batch_labels in train_loader:
             
        D_pred = model(batch_data, strain)
        loss = loss_function(batch_labels, D_pred, Y_train_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    ep_time_1 = time.time()
    # Validation step
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients during validation
        val_losses = []
        for batch_data, batch_labels in val_loader:
            D_test_pred = model(batch_data, strain)
            test_loss = loss_function(batch_labels, D_test_pred, Y_test_weight)
            val_losses.append(test_loss.cpu().data.numpy())
        avg_val_loss = np.mean(val_losses)

    l_train = loss.cpu().data.numpy()  # Use the loss from the last batch as training loss for this epoch

    train_loss_ls.append(l_train)
    test_loss_ls.append(avg_val_loss)

    if avg_val_loss < lowest_loss:
        best_model_id = epoch
        lowest_loss = avg_val_loss
        torch.save(model.state_dict(), 'Params/PCA_DON_'+str(epoch)+'.pt')

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, train loss: {l_train:.3e}, val loss: {avg_val_loss:.3e}, best model: {best_model_id}, lowest error: {lowest_loss:.3e}, elapsed time: {time.time() - begin_time:.3e}s, epoch time: {ep_time_1 - ep_time_0:.3e}s ")
print('################### done #################################')


