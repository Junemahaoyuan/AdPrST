
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


class Regularizer(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Regularizer, self).__init__()
        self.den1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.den1.bias.data.fill_(0.0)
        self.den1.weight.data = torch.normal(0.0, 0.001, [hidden_dim1, input_dim])
        self.den2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.den2.bias.data.fill_(0.0)
        self.den2.weight.data = torch.normal(0.0, 0.001, [hidden_dim2, hidden_dim1])
        self.output = torch.nn.Linear(hidden_dim2, 1)
        self.output.bias.data.fill_(0.0)
        self.output.weight.data = torch.normal(0.0, 0.001, [1,hidden_dim2])
        self.act = torch.sigmoid
    def forward(self, inputs):
        dc_den1 = self.act(self.den1(inputs))
        dc_den2 = torch.sigmoid((self.den2(dc_den1)))
        output = self.output(dc_den2)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.linear_pl = nn.Linear(64, n_h)  
        self.linear_mi = nn.Linear(3000, n_h)  
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        
        if h_pl.shape[1] != 64:
            h_pl = h_pl[:, :64]  
        
        if h_mi.shape[1] != 3000:
            # Extend h_mi to match the expected dimension of 3000
            if h_mi.shape[1] < 3000:
                padding_size = 3000 - h_mi.shape[1]
                h_mi = F.pad(h_mi, (0, padding_size), 'constant', 0)  
            elif h_mi.shape[1] > 3000:
                h_mi = h_mi[:, :3000]  # Slice to 3000 features if it exceeds

        
        h_pl = self.linear_pl(h_pl)  
        h_mi = self.linear_mi(h_mi)  

        c_x = c.expand_as(h_pl)  
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)


        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits



class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T 
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1)    

class Attention(nn.Module):
    def __init__(self,in_size):
        super(Attention, self).__init__()
        self.project=nn.Linear(in_size, 1, bias=False) 

    def forward(self,z):
        w = self.project(z)
        beta = torch.softmax(w,dim=1)
        return (beta*z).sum(1),beta  




class AdversarialFramework(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu, esparse=False):
        super(AdversarialFramework, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.esparse = esparse
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)  
        self.regularizer = Regularizer(self.out_features, 64, 32)  
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj, graph_neigh):
        
        z = F.dropout(feat, self.dropout, self.training)  
        z = torch.mm(z, self.weight1)
        if self.esparse:
            z = torch.spmm(adj, z)
        else:
            z = torch.mm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        
        emb = self.act(z)

        
        f_z = self.regularizer(emb)  
        r = torch.normal(0.0, 1.0, [feat.shape[0], self.out_features]).cuda()  
        f_r = self.regularizer(r)  
        reg_loss = -f_r.mean() + f_z.mean()  

        
        g = self.read(emb, graph_neigh) 
        g = self.sigm(g)  

        ret = self.disc(g, emb, feat_a)  
        ret_a = self.disc(g, feat_a, emb)  

        return hiden_emb, h, ret, ret_a, reg_loss  


class FusionAdversarialFramework(torch.nn.Module):
    def __init__(self, in_features, out_features, esparse=False):
        super(FusionAdversarialFramework, self).__init__()

        self.attention1 = Attention(out_features)
        self.attention2 = Attention(in_features)
        self.attention3 = Attention(2)
        self.adprst = AdversarialFramework(in_features, out_features, esparse)

    def forward(self, features, features_a, adj1, adj2, graph_neigh1, graph_neigh2):
       
        hiden_emb1, h1, ret1, ret_a1, reg_loss1 = self.adprst(features, features_a, adj1, graph_neigh1)
        hiden_emb2, h2, ret2, ret_a2, reg_loss2 = self.adprst(features, features_a, adj2, graph_neigh2)       
        hiden_emb = torch.stack([hiden_emb1, hiden_emb2], dim=1)
        h = torch.stack([h1, h2], dim=1)
        ret = torch.stack([ret1, ret2], dim=1)
        ret_a = torch.stack([ret_a1, ret_a2], dim=1)
   
        hiden_emb, _ = self.attention1(hiden_emb)
        h, _ = self.attention2(h)
        ret, _ = self.attention3(ret)
        ret_a, _ = self.attention3(ret_a)
        
        total_reg_loss = reg_loss1 + reg_loss2

        return hiden_emb, h, ret, ret_a, total_reg_loss


