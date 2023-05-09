import torch.nn as nn
import torch
import torch.nn.functional as F
class MLP(nn.Module):
  def __init__(self,in_channel,hidden_dim, out_channel):
    super().__init__()
    self.embedding = nn.Embedding(6,44)

    # MLP의 기본이 되는 fc layer 4개를 구상합니다.
    self.Linear1 = nn.Linear(in_channel,hidden_dim[0])   
    self.Linear2 = nn.Linear(hidden_dim[0],hidden_dim[1])
    self.Linear3 = nn.Linear(hidden_dim[1]+11,hidden_dim[2])
    self.Linear4 = nn.Linear(hidden_dim[2],out_channel)
    self.bn1 =  nn.BatchNorm1d(hidden_dim[0])
    self.bn2 =  nn.BatchNorm1d(hidden_dim[1])
    self.bn3 =  nn.BatchNorm1d(hidden_dim[2])




  def forward(self,x,y):
    # batch size를 제외한 차원을 flatten 시켜줍니다. 
    x = x.long()
    x = self.embedding(x)
        
    
    x = torch.reshape(x,(x.size(0),-1))
    
    
    # 한개의 layer마다 fc layer와 relu function 을 쌓아줍니다.
    

    x = self.Linear1(x)
    x = F.relu(x)
    # x = self.bn1(x)
    x = self.Linear2(x)
    x = F.relu(x)

    # x = self.bn2(x)
    x = torch.concat([x,y],dim =1)
    x = self.Linear3(x)
    x = F.relu(x)
    x = self.Linear4(x)
    
    
    
    return x

