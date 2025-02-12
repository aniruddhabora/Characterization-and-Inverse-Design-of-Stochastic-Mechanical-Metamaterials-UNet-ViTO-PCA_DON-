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
        print('brach 01: ', coefficients_01.shape)
        #print(coefficients_02.shape)
        #print(coefficients_03.shape)

        print('pca 01 shape:',self.pca_basis_01.shape)

        #sys.exit()
        # Use the coefficients with the fixed PCA basis
        # Assuming the pca_basis is shaped [output_dim, num_basis] for matrix multiplication compatibility
        # and coefficients are shaped [batch_size, num_basis]
        output_01 = torch.matmul(coefficients_01, self.pca_basis_01) + self.mean_function_01
        output_02 = torch.matmul(coefficients_02, self.pca_basis_02) + self.mean_function_02
        output_03 = torch.matmul(coefficients_03, self.pca_basis_03) + self.mean_function_03

        print('output shape:', output_01.shape)
        sys.exit()
        #out = self.output_layer(output)
        #print(out.shape)
        #sys.exit()
        out = torch.stack((output_01, output_02, output_03), dim=1)
        out = torch.reshape(out,(out.shape[0],out.shape[1], 2 ,51))
        out = out*y[None,None,None,:]
        #print(out.shape)
        #sys.exit()

        return out

    def loss(self, true, pred, weights):

        true = 2*(true-self.par['out_shift'])/self.par['out_scale'] - 1.0
        pred = 2*(pred-self.par['out_shift'])/self.par['out_scale'] - 1.0
        
        loss_value = (torch.mean(weights*(pred - true)**2))/(torch.mean(true**2))
 

        return loss_value

'''

class DONForClassification(PCAFixedTrunkNet):
    def __init__(self, par, num_classes):
        super(DONForClassification, self).__init__(par)
        n_channels = self.par['out_ch']

        # Assuming 'n_channels' corresponds to the output channels of the last decoder block
        # Global Average Pooling to reduce each feature map to a single number
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        # The number of output features of global_avg_pool is the same as the number of output channels
        # of the last decoder block. Adjust 'out_features' to match the number of classes for classification.
        self.classifier = nn.Linear(n_channels, num_classes)

    def forward(self, x):
        # Use the original forward pass up to the last decoder block
        out = super().forward(x)

        # Apply global average pooling to the output of the last decoder block
        out = self.global_avg_pool(out)

        # Flatten the output for the classifier
        out = torch.flatten(out, 1)

        # Classifier to predict class probabilities
        out = self.classifier(out)

        return out
'''        
