import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class BranchNet(nn.Module):
    def __init__(self, num_channels, nx, ny, num_basis):
        super(BranchNet, self).__init__()
        # Calculate the size of the flattened features after convolutions and pooling
        # This requires knowing the input size and the architecture
        self.num_channels = num_channels
        self.nx = nx
        self.ny = ny
        self.num_basis = num_basis
        
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the feature maps after pooling layers
        # Placeholder for the size of the feature map before the fully connected layer
        feature_size_x = nx // 32  # Assuming two pooling layers
        feature_size_y = ny // 32
        
        self.fc = nn.Sequential(nn.Linear(128 * feature_size_x * feature_size_y, num_basis*3, bias=True),
                  nn.ReLU(),
                  nn.Linear( num_basis*3, num_basis*3, bias=True),
                  nn.Tanh(),
                  nn.Linear( num_basis*3, num_basis*3, bias=True),
                  nn.ReLU(),
                  nn.Linear( num_basis*3, num_basis*3, bias=True))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.tanh(self.conv3(x))
        x = self.pool(x)
        x = F.tanh(self.conv4(x))
        x = self.pool(x)
        x = F.tanh(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten the features for the fully connected layer
        x = self.fc(x)
        
        return x


class PCAFixedTrunkNet(nn.Module):
    def __init__(self, par, branch_net, pca_basis_01, mean_function_01, pca_basis_02, mean_function_02, pca_basis_03, mean_function_03):
        super(PCAFixedTrunkNet, self).__init__()
        
        self.par = par
        self.branch_net = branch_net
             
        # PCA basis and mean function are fixed parameters, not trained

        self.pca_basis_01 = nn.Parameter(pca_basis_01, requires_grad=False)
        self.mean_function_01 = nn.Parameter(mean_function_01, requires_grad=False)

        self.pca_basis_02 = nn.Parameter(pca_basis_02, requires_grad=False)
        self.mean_function_02 = nn.Parameter(mean_function_02, requires_grad=False)

        self.pca_basis_03 = nn.Parameter(pca_basis_03, requires_grad=False)
        self.mean_function_03 = nn.Parameter(mean_function_03, requires_grad=False)

        self.output_layer = nn.Linear(pca_basis_01.shape[1], 51*2) 

    def forward(self, branch_input, y):
        # Compute coefficients for the PCA basis using the branch network
        coefficients = self.branch_net(branch_input)
        coefficients_01 = coefficients[:,0:62]
        coefficients_02 = coefficients[:,62:62+62]
        coefficients_03 = coefficients[:,62+62:62+62+62]
        
        # Use the coefficients with the fixed PCA basis
        # Assuming the pca_basis is shaped [output_dim, num_basis] for matrix multiplication compatibility
        # and coefficients are shaped [batch_size, num_basis]
        output_01 = torch.matmul(coefficients_01, self.pca_basis_01) + self.mean_function_01
        output_02 = torch.matmul(coefficients_02, self.pca_basis_02) + self.mean_function_02
        output_03 = torch.matmul(coefficients_03, self.pca_basis_03) + self.mean_function_03

        out = torch.stack((output_01, output_02, output_03), dim=1)
        out = torch.reshape(out,(out.shape[0],out.shape[1], 2 ,51))
        out = out*y[None,None,None,:]
        
        return out

    def loss(self, true, pred, weights):

        true = 2*(true-self.par['out_shift'])/self.par['out_scale'] - 1.0
        pred = 2*(pred-self.par['out_shift'])/self.par['out_scale'] - 1.0
        
        loss_value = (torch.mean(weights*(pred - true)**2))/(torch.mean(true**2))
 

        return loss_value

