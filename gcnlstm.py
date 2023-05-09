
import pandas as pd 
import numpy as np
from torch import nn 
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import LSTM



class lstmf(nn.Module):
    def __init__ (self, input_dim, hidden_dim,layers):
        super().__init__()
        self.lstm  = LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers=layers,
                            dropout = 0.2, bidirectional = True,
                            batch_first=True)
        
        self.fc = nn.Linear(2*21*hidden_dim,300)
        self.fc2 = nn.Linear(300,300)
        self.fc3 = nn.Linear(300,1)

    def forward(self,x):
        
        out,(a,b) = self.lstm(x)
        out = torch.flatten(out,start_dim=1)
        
        return out




class GCNLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, n_atom, act=None, bn=False):
        super(GCNLayer, self).__init__()
        
        self.use_bn = bn
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.bn = nn.BatchNorm1d(n_atom)
        self.activation = act
        
    def forward(self, x, adj):
        
        out = self.linear(x) 
        out = torch.matmul(adj, out)
        if self.use_bn:
            out = self.bn(out)
        if self.activation != None:
            out = self.activation(out)
        return out, adj



class SkipConnection(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        out = in_x + out_x
        return out




class GatedSkipConnection(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(GatedSkipConnection, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_coef_in = nn.Linear(out_dim, out_dim)
        self.linear_coef_out = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        z = self.gate_coefficient(in_x, out_x)
        out = torch.mul(z, out_x) + torch.mul(1.0-z, in_x)
        return out
            
    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)

        return self.sigmoid(x1+x2)



class GCNBlock(nn.Module):
    
    def __init__(self, n_layer,n_layer2,n_layer3, in_dim, hidden_dim, out_dim, n_atom, bn=True, sc='gsc'):
        super(GCNBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer-1 else hidden_dim,
                                        n_atom,
                                        nn.ReLU() if i!=n_layer-1 else None,
                                        bn))
        self.layers2 = nn.ModuleList()
        for i in range(n_layer2):
            self.layers2.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer2-1 else hidden_dim,
                                        n_atom,
                                        nn.ReLU() if i!=n_layer2-1 else None,
                                        bn))
        self.layers3 = nn.ModuleList()
        for i in range(n_layer3):
            self.layers3.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer3-1 else hidden_dim,
                                        n_atom,
                                        nn.ReLU() if i!=n_layer3-1 else None,
                                        bn))
        self.relu = nn.ReLU()
        if sc=='gsc':
            self.sc = GatedSkipConnection(in_dim, out_dim)
        elif sc=='sc':
            self.sc = SkipConnection(in_dim, out_dim)
        elif sc=='no':
            self.sc = None
        else:
            assert False, "Wrong sc type."
        
    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            out, adj = layer((x if i==0 else out), adj)
        
        for i, layer in enumerate(self.layers2):
            out2, adj = layer((x if i==0 else out), adj)
            
        for i, layer in enumerate(self.layers3):
            out3, adj = layer((x if i==0 else out), adj)
        out = (out+out2+out3)/3
  
        if self.sc != None:
            out = self.sc(residual, out)
        out = self.relu(out)

        return out, adj

class ReadOut(nn.Module):
    
    def __init__(self, in_dim, out_dim, act=None):
        super(ReadOut, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim= out_dim 
        self.linear = nn.Linear(self.in_dim, 
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
        
        out = torch.sum(out, 1)
        
        if self.activation != None:
            out = self.activation(out)
        return out

class Predictor(nn.Module):
    
    def __init__(self, in_dim, out_dim, act=None):
        super(Predictor, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act
        
    def forward(self, x):
        out = self.linear(x)
        if self.activation != None:
            out = self.activation(out)
        return out

class lstmGCNNetincepimb(nn.Module):
    
    def __init__(self, args):
        super(lstmGCNNetincepimb, self).__init__()
        self.embedding = nn.Embedding(6,44)
        
        self.blocks = nn.ModuleList()
        for i in range(args.n_block):
            self.blocks.append(GCNBlock(args.n_layer,args.n_layer2, args.n_layer3,
                                        args.in_dim if i==0 else args.hidden_dim,
                                        args.hidden_dim,
                                        args.hidden_dim,
                                        args.n_atom,
                                        args.bn,
                                        args.sc))
        self.lstm = lstmf(44,80,1)
        self.readout = ReadOut(args.hidden_dim, 
                               args.pred_dim,
                               act=nn.ReLU())
        self.pred1 = Predictor(args.pred_dim1,
                               args.pred_dim2,
                               act=nn.ReLU())
        self.pred2 = Predictor(args.pred_dim2,
                               args.pred_dim3,
                               act=nn.Tanh())
        self.pred3 = Predictor(args.pred_dim3,
                               args.out_dim)
       

        self.fc3 = nn.Linear(3360,100)
        self.fc4 = nn.Linear(100,100)
        self.bn1 = nn.BatchNorm1d(50)
        self.bn2 = nn.BatchNorm1d(25)
        
        
        
    def forward(self, x,y, adj):
        
        
        x = x.long()
        x = self.embedding(x)
        for i, block in enumerate(self.blocks):
            out, adj = block((x if i==0 else out), adj)
        out = self.readout(out)
        out2 = self.lstm(x)
        out2 = self.fc3(out2)    
        out2 = F.selu(out2) 
        out2 = self.fc4(out2)     
        out1 = torch.cat((out,out2,y),dim =1)   
        out = self.pred1(out1)
        out = self.pred2(out)
        out = self.pred3(out)

        return out


