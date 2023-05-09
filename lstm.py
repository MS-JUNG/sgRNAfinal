from torch import nn 
from torch.nn import LSTM 
import torch
import torch.nn.functional as F

class Lstmfc(nn.Module):
    def __init__ (self, input_dim, hidden_dim,layers):
        super().__init__()
        self.embedding = nn.Embedding(6,44)
        self.lstm  = LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers=layers,
                          bidirectional = True,
                            batch_first=True)
        
        self.fc = nn.Linear(3371,300)

        self.fc2 = nn.Linear(300,100)
        self.fc3 = nn.Linear(100,1)

    def forward(self,x,y):
        x = x.long()
             
        x = self.embedding(x)

        
        out,(a,b) = self.lstm(x)
        
        out = torch.flatten(out,start_dim=1)
        
        out = torch.concat([out,y],dim = 1)   
        out = self.fc(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        

        return out
    