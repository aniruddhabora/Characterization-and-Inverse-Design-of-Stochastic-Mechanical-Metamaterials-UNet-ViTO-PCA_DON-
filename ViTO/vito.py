import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F


class Conv_Block(nn.Module):
    def __init__(self,in_c,out_c,k,num_groups=1):
        super(Conv_Block, self).__init__()
        
        self.block = nn.Sequential(
                     nn.Conv2d(in_c, out_c, k, padding='same'),
                     nn.GroupNorm(num_groups,out_c),
                     nn.GELU())

    def forward(self, x):
        return self.block(x)



class S_Attn(nn.Module):
    def __init__(self, in_c, n_heads=4, dim_h =32 ):
        super(S_Attn, self).__init__()
        
        self.conv_attn = nn.Conv2d(in_c,3*n_heads*dim_h, kernel_size=1)
        self.n_heads = n_heads
        self.dim_h = dim_h
        self.conv_h = nn.Sequential(nn.Conv2d(n_heads*dim_h, in_c, kernel_size=1),
                      nn.GroupNorm(1,in_c))

    def forward(self, x):

        # x = [B,C,X,Y]
        B, _, X, Y = x.shape
        H = self.n_heads
        C = self.dim_h
        temp = self.conv_attn(x)   # [B,3CH,X,Y] , H = n_heads
        temp = temp.reshape(B,3,H,C,X*Y)
        
        Q = temp[:,0]    # [B,H,C,X*Y]
        K = temp[:,1]    # [B,H,C,X*Y]
        V = temp[:,2]    # [B,H,C,X*Y]

        corr = torch.einsum('bhci,bhcj->bhij', Q, K)     # spatial attention (QKT) [B,H,X*Y,X*Y]
        D1 = torch.tensor(C, dtype=torch.int)
        D2 = torch.tensor(X*Y, dtype=torch.int)
        corr = corr/torch.sqrt(D1*D2)                      # normalization
        
        out = torch.einsum('bhij,bhcj->bhci',corr, V)    # like Expectation (weighted sum) [B,H,C,X*Y]
        out = out.reshape(B,H*C,X,Y)
        out = self.conv_h(out)                           # [B,C,X,Y]

        return out


class C_Attn(nn.Module):
    def __init__(self, in_c, n_heads=4, dim_h =32):
        super(C_Attn, self).__init__()

        self.conv_attn = nn.Conv2d(in_c,3*in_c*n_heads, kernel_size=1)
        self.n_heads = n_heads
        self.dim_h = dim_h
        self.conv_h = nn.Sequential(nn.Conv2d(in_c*n_heads, in_c, kernel_size=1),
                      nn.GroupNorm(1,in_c)) 

    def forward(self, x):

        # x = [B,C,X,Y]
        B, _, X, Y = x.shape
        H = self.n_heads
        C = self.dim_h
        temp = self.conv_attn(x)                       # [B,3CH,X,Y] , H = n_heads
        temp = temp.reshape(B,3,H,C,X*Y)

        Q = temp[:,0]    # [B,H,C,X*Y]
        K = temp[:,1]    # [B,H,C,X*Y]
        V = temp[:,2]    # [B,H,C,X*Y]

        corr = torch.einsum('bhid,bhjd->bhij', Q, K)   # channel attention (QKT) [B,H,C,C]
        D1 = torch.tensor(X*Y, dtype=torch.int)
        D2 = torch.tensor(C, dtype=torch.int)
        corr = corr/torch.sqrt(D1*D2)                      # normalization        

        out = torch.einsum('bhij,bhid->bhjd',corr, V)  # like Expectation (weighted sum) [B,H,C,X*Y]
        out = out.reshape(B,H*C,X,Y)
        out = self.conv_h(out)                         # [B,C,X,Y]

        return out



class Enc_Block(nn.Module):
    def __init__(self,in_c,out_c,k, num_groups=1):
        super(Enc_Block, self).__init__()

        self.conv_block = Conv_Block(in_c,out_c,k, num_groups)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
            
    def forward(self, x):

        out = self.conv_block(x)
        out = self.mp(out)

        return out


class Dec_Block(nn.Module):
    def __init__(self,in_c,out_c,k,flag_s_attn=False, flag_c_attn=False, num_groups=1):
        super(Dec_Block, self).__init__()
        
        self.convtrans = nn.ConvTranspose2d(in_c,out_c,kernel_size=2,stride=2)
        self.conv_block = Conv_Block(2*out_c,out_c,k,num_groups)
        
        if flag_s_attn==True:
            self.s_attn = S_Attn(out_c)
        else:
            self.s_attn = nn.Identity()

        if flag_c_attn==True:
            self.c_attn = C_Attn(out_c)
        else:
            self.c_attn = nn.Identity()

 
    def forward(self,d,e):

        # d: [B,in_c,X,Y]
        # e: [B,out_c,2X,2Y]

        e = self.c_attn(e)
        e = self.s_attn(e)
        d = self.convtrans(d)             # [B,out_c,2X,2Y]
        temp = torch.cat([d,e],dim=1)     # [B,2*out_c,2X,2Y]
        out = self.conv_block(temp)       # [B,out_c,2X,2Y]
 
        return out                        

        

class UNET(nn.Module):
    def __init__(self,par):
        super(UNET, self).__init__()

        self.par = par
        
        n_channels = self.par['n_channels']
        k = self.par['k']

        self.enc0 = Conv_Block(self.par['inp_ch'], n_channels, k )                                  # [B,C,X,Y]
        
        self.enc1 = Enc_Block(n_channels, 2*n_channels, k)                                          # [B,2C,X/2,Y/2]
        self.enc2 = Enc_Block(2*n_channels, 4*n_channels, k)                                        # [B,4C,X/4,Y/4]
        self.enc3 = Enc_Block(4*n_channels, 8*n_channels, k)                                        # [B,8C,X/8,Y/8]

        self.s_attn = S_Attn(8*n_channels)                                                          # [B,8C,X/8,Y/8]

        self.dec3 = Dec_Block(8*n_channels,4*n_channels, k, flag_s_attn=False, flag_c_attn=False)    # [B, 4C,X/4,Y/4]
        self.dec2 = Dec_Block(4*n_channels,2*n_channels, k, flag_s_attn=False, flag_c_attn=False)    # [B, 2C, X/2, Y/2]
        self.dec1 = Dec_Block(2*n_channels,n_channels, k, flag_s_attn=False, flag_c_attn=False)     # [B, C, X, Y]
       
        self.dec0 = nn.Sequential(nn.GroupNorm(int(n_channels/4), n_channels),
                                  nn.Conv2d(n_channels,self.par['out_ch'],kernel_size=1))

        self.final_lin_0 = nn.Sequential(nn.AdaptiveAvgPool2d((4, 32)),
                           nn.Linear(32,51))#,
        
        self.final_lin_1 = nn.Linear(4,2)


    def forward(self, x, y):
        # [B, inp_ch,X,Y]

        x = 2*(x-self.par['inp_shift'])/(self.par['inp_scale'])-1

        e0 = self.enc0(x)           # [B,C,X,Y]
        e1 = self.enc1(e0)          # [B,2C,X/2,Y/2]
        e2 = self.enc2(e1)          # [B,4C,X/4,Y/4]
        e3 = self.enc3(e2)          # [B,8C,X/8,Y/8]

        temp = self.s_attn(e3)      # [B,8C,X/8,Y/8]
        
        d3 = self.dec3(temp, e2)    # [B, 4C,X/4,Y/4]
        d2 = self.dec2(d3, e1)      # [B, 2C, X/2,Y/2]
        d1 = self.dec1(d2, e0)      # [B,C, X, Y]

        out = self.dec0(d1)         # [B, out_ch, X, Y]
        
        out = self.final_lin_0(out)
        out = out.permute(0,1,3,2)
        out = self.final_lin_1(out)
        out = out.permute(0,1,3,2)
        out = 0.5*(out+1)*self.par['out_scale'] + self.par['out_shift']
        out = out*y[None,None,None,:]

        return out



    def loss(self, true, pred, weights):
        
        true = 2*(true-self.par['out_shift'])/self.par['out_scale'] - 1.0
        pred = 2*(pred-self.par['out_shift'])/self.par['out_scale'] - 1.0

        #loss_value = torch.norm( true-pred, p=2 )/(10**-5 + torch.norm( true, p=2 ) )
        loss_value = (torch.mean(weights*(pred - true)**2))/(torch.mean(true**2)) 

        return loss_value











