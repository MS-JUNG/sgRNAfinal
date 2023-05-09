from torch import nn
import math
import torch





class Lstmfc(nn.Module):
    def __init__ (self, input_dim, hidden_dim,layers):
        super().__init__()
        self.lstm  = LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers=layers,
                            dropout = 0.1, bidirectional = True,
                            batch_first=True)
        
        self.fc = nn.Linear(2*21*hidden_dim,300)
        self.fc2 = nn.Linear(300,300)
        self.fc3 = nn.Linear(300,1)

    def forward(self,x):


        out,(a,b) = self.lstm(x)
        out = torch.flatten(out,start_dim=1)
        return out

    

import torch 
import torch.nn as nn


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
        # if self.activation != None:
        #     out = self.activation(out)
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
    
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, n_atom, bn=True, sc='gsc'):
        super(GCNBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer-1 else hidden_dim,
                                        n_atom,
                                        
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
        hidden = []
        for i, layer in enumerate(self.layers):
            out1, adj = layer((x if i==0 else out1), adj)
            

            
            
       
            
        
        
            
        # hidden = hidden.tolisy()
        # [t.size() for t in hidden]
        # breakpoint()
        # if self.sc != None:
        #     out1 = self.sc(residual, out1)
        # if self.sc != None:
        #     out2 = self.sc(residual, out2)
        # out = self.relu(out)
        return out1


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        
        return self.dropout(x)
encoder  = nn.TransformerEncoderLayer(d_model=64, nhead=8)
 
class transripr(nn.Module):

   def __init__ (self, input_dim, hidden_dim,fc_dim):
        super().__init__()
        self.embedding = nn.Embedding(6,input_dim)
        self.conv1 = nn.Conv1d(input_dim,hidden_dim,3,1,1)
        self.pool = nn.AvgPool1d(hidden_dim, stride=1, padding=0)
        self.position = PositionalEncoding(hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, batch_first = True)
        self.encoder  = nn.TransformerEncoder(encoder, num_layers=6)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(2688,fc_dim[0])
                                
        self.fc = nn.Sequential(
                                nn.Linear(fc_dim[0]+11,fc_dim[1]),
                                nn.Tanh(),
                                nn.Linear(fc_dim[1],fc_dim[2]),
                                nn.Tanh(),
                                nn.Linear(fc_dim[2],fc_dim[3]),
                                )
        self.gcn = GCNBlock(1,44,64,64,21,True,'gsc')
        
      

   def forward(self,x,adj,y):
     x= x.long()     
     x = self.embedding(x)
     x1 = x.transpose(1,2)
     x1 =  self.conv1(x1)   
     out1 = self.gcn(x,adj)
     pos1 = self.position(out1) 
     x = self.encoder(pos1)
     x1 = x1.transpose(1,2)    
     x = torch.concat([x,x1], dim = 1)
     x  = torch.flatten(x,start_dim=1)
     x =self.fc2(x)
     x = torch.concat([x,y],dim =1)
     x = self.fc(x)

     return x


     



      
