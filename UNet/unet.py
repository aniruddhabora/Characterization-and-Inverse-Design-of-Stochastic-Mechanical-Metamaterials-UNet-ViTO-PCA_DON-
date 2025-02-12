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
        d = self.convtrans(d)             
        temp = torch.cat([d,e],dim=1)     
        out = self.conv_block(temp)       
 
        return out                        

        

class UNET(nn.Module):
    def __init__(self,par):
        super(UNET, self).__init__()

        self.par = par
        
        n_channels = self.par['n_channels']
        k = self.par['k']

        self.enc0 = Conv_Block(self.par['inp_ch'], n_channels, k )                                  
        
        self.enc1 = Enc_Block(n_channels, 2*n_channels, k)                                          
        self.enc2 = Enc_Block(2*n_channels, 4*n_channels, k)                                        
        self.enc3 = Enc_Block(4*n_channels, 8*n_channels, k)                                        

        self.dec3 = Dec_Block(8*n_channels,4*n_channels, k, flag_s_attn=False, flag_c_attn=False)    
        self.dec2 = Dec_Block(4*n_channels,2*n_channels, k, flag_s_attn=False, flag_c_attn=False)   
        self.dec1 = Dec_Block(2*n_channels,n_channels, k, flag_s_attn=False, flag_c_attn=False)     
       
        self.dec0 = nn.Sequential(nn.GroupNorm(int(n_channels/4), n_channels),
                                  nn.Conv2d(n_channels,self.par['out_ch'],kernel_size=1))

        self.final_lin_0 = nn.Sequential(nn.AdaptiveAvgPool2d((4, 32)),
                           nn.Linear(32,31))#,
        
        self.final_lin_1 = nn.Linear(4,2)


    def forward(self, x, y):

        x = 2*(x-self.par['inp_shift'])/(self.par['inp_scale'])-1

        e0 = self.enc0(x)           
        e1 = self.enc1(e0)          
        e2 = self.enc2(e1)          
        e3 = self.enc3(e2)          

        temp = e3   
        
        d3 = self.dec3(temp, e2)    
        d2 = self.dec2(d3, e1)      
        d1 = self.dec1(d2, e0)      

        out = self.dec0(d1)         
        
        out = F.interpolate(out, size=(51, 1), mode='bilinear', align_corners=False)
        out = out.squeeze(-1)

        out = torch.reshape(out,(out.shape[0],3,2,out.shape[2]))
        out = 0.5*(out+1)*self.par['out_scale'] + self.par['out_shift']
        out = out*y[None,None,None,:]

        return out



    def loss(self, true, pred, weights):
        
        true = 2*(true-self.par['out_shift'])/self.par['out_scale'] - 1.0
        pred = 2*(pred-self.par['out_shift'])/self.par['out_scale'] - 1.0

        #loss_value = torch.norm( true-pred, p=2 )/(10**-5 + torch.norm( true, p=2 ) )
        loss_value = (torch.mean(weights*(pred - true)**2))/(torch.mean(true**2)) 

        return loss_value











