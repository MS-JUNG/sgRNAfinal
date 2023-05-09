from gcnlstm import lstmGCNNetincepimb
import easydict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from torch import nn 
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import LSTM
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import pickle




with open(file='wt_seq_data_array.pkl', mode='rb') as f:
    data = pickle.load(f)

### ViennaRNA 활용을 위해 숫자 데이터로 A, T, G, C 로 변환합니다
data_base = list()
for i in range(55604):
    seq = ''
    for j in range(21):
    
        if data[0][i][j] == 2:
            seq = seq +'A'
        elif data[0][i][j] == 3:
            seq = seq +'T'
        elif data[0][i][j] == 4:
            seq = seq +'C'
        elif data[0][i][j] == 5:
            seq = seq +'G'
    data_base.append(seq)


class sgrnadataset(Dataset): 
  def __init__(self,seq,adjacency,bio,wt_target):
    self.seq = seq
    self.target = wt_target
    self.adj = adjacency
    self.bio = bio
  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.target)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    x = torch.FloatTensor(self.seq[idx])    
    z = torch.FloatTensor(self.adj[idx])
    k = self.bio[idx]
    y  = self.target[idx]

    return x,z,k,y
new_arr = np.delete(data[0], 0, axis=1)



data_ad = []
with open("/home/jovyan/Minsu/sgRNA/sgRNA/wt_adj.txt", "r") as f:
  example = f.readlines() 

  for line in example:
    if len(line)>28:  
        data_ad.append(line[:21])
    else:
            pass

adjacency = []

for i in range(len(data_ad)):
    vin = data_ad[i]
    con = np.zeros((21,21))
    

    for j in range(21):     
        con[j][j] = con[j][j] +1
        if j ==0:
            con[j][j+1] =  con[j][j+1] +1 
            
        elif j == 20:
            con[j][j-1] =  con[j][j-1] +1 

        else :
            con[j][j+1] =  con[j][j+1] +1 
            con[j][j-1] =  con[j][j-1] +1 


    stack = []
    index = []

    for k in range(21):
        
        if vin[k] == '.':
            pass
        elif vin[k] == '(':
            index.append (k)
            stack.append('(')
        elif vin[k] == ')':
            stack.pop()
            l = index.pop()
            
            con[l][k] = con[l][k] +1 
            con[k][l] = con[k][l] +1 
         
    adjacency.append(con)


### 데이터를 스플릿 합니다.
wt_base = new_arr
wt_target = data[2]
wt_bio = data[1]
wt_adj = adjacency 

feature_train, feature_test, adj_train, adj_test, bio_train, bio_test, efficiency_train, efficiency_test = train_test_split(wt_base,wt_adj, wt_bio,
                             wt_target, 
                             test_size = 0.15, 
                             random_state = 42,
                             shuffle = True)
feature_train, feature_val, adj_train, adj_val,bio_train,bio_val, efficiency_train, efficiency_val = train_test_split(feature_train, adj_train,bio_train, efficiency_train, 
                         test_size = 0.1, 
                         random_state = 42,
                         shuffle =True
                      )
train_set = sgrnadataset(feature_train,adj_train,bio_train,efficiency_train)
trainloader = DataLoader(train_set, batch_size= 64, shuffle = True)
val_set = sgrnadataset(feature_val,adj_val,bio_val,efficiency_val)
valloader = DataLoader(val_set, batch_size= 64, shuffle = True)
test_set =  sgrnadataset(feature_test,adj_test,bio_test,efficiency_test)
testloader = DataLoader(test_set, batch_size= 64, shuffle = True)
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



args = easydict.EasyDict({
 
       "n_layer" : 3,
       "n_layer2" : 2,
       "n_layer3"  : 1,
       "in_dim"   : 44,
       "hidden_dim" : 30,
       "n_block" : 1,
       "n_atom" : 21,
       "bn" : True,
       "sc" : 'gsc',
       'pred_dim' : 50,
       'pred_dim1' : 161,
       'pred_dim2' : 150,
       'pred_dim3' : 150,
       'out_dim' : 1,
 })


net = lstmGCNNetincepimb(args)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.0005)
criterion = nn.MSELoss()
net.cuda()

epoch = 1


val_min = []
min_loss = +10000
v_epoch= 0
for i in tqdm(range(epoch)):

    net.train()
    train_loss = 0
    v_epoch +=1
    for  data in trainloader:
            
            optimizer.zero_grad() 

            # get the inputs
            list_feature, list_adj,list_bio, cas_efficiency = data
            list_feature = list_feature.cuda().float()
            list_adj = list_adj.cuda().float()
            list_bio = list_bio.cuda().float()
                
            cas_efficiency = cas_efficiency.view(-1, 1)
            cas_efficiency =  cas_efficiency.cuda().float()
            outputs= net(list_feature,list_bio,list_adj)
            
        
            loss = criterion(outputs, cas_efficiency)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
    train_loss = train_loss / len(trainloader)


    net.eval()
    
    val_loss = 0 
    with torch.no_grad():
                                 
        for data in valloader:
                
                list_feature, list_adj,list_bio, cas_efficiency = data
                list_feature = list_feature.cuda().float()
                list_adj = list_adj.cuda().float()
                list_bio = list_bio.cuda().float()                        
                cas_efficiency = cas_efficiency.view(-1, 1)
                cas_efficiency =  cas_efficiency.float().cuda()  
                outputs= net(list_feature,list_bio,list_adj)                       
                loss = criterion(outputs, cas_efficiency)
                val_loss += loss.item()

        val_loss = val_loss / len(valloader)
        val_min.append(val_loss)

        if min_loss > min(val_min):
            min_loss = val_loss
            torch.save(net.state_dict(),'/home/jovyan/Minsu/sgRNA/sgRNA/lstmgcn.pt')




    
    print('Epoch {}, Loss(train/val) {:2.4f}/{:2.4f}'.format(v_epoch, train_loss, val_loss))





    with torch.no_grad():
            net = lstmGCNNetincepimb(args)
            net.load_state_dict(torch.load('lstmgcn.pt'))
            net.cuda()
            efficiency_total = list()
            pred_efficiency_total = list()

            for data in testloader:

                list_feature, list_adj,list_bio, cas_efficiency = data
                list_feature = list_feature.cuda().float()
                list_adj = list_adj.cuda().float()
                list_bio = list_bio.cuda().float()
                cas_efficiency = cas_efficiency.view(-1, 1)
                cas_efficiency =  cas_efficiency.float().cuda()  
                outputs= net(list_feature,list_bio,list_adj)
                efficiency_total += cas_efficiency.tolist()
                cas_efficiency = cas_efficiency.view(-1, 1)          
                pred_efficiency_total += outputs.view(-1).tolist()

            
            Spear = stats.spearmanr(efficiency_total, pred_efficiency_total)
            mse = mean_squared_error(efficiency_total, pred_efficiency_total)
           
                
            print("GCNLSTM: ",Spear)
            print("GCNLSTM: ",mse)


from transcrip import transripr

net = transripr(44,64,[500,250,125,1])
optimizer = torch.optim.Adamax(net.parameters())
val_min = []
min_loss = +10000
v_epoch = 0
net.cuda()

for i in tqdm(range(epoch)):
    net.train()
    train_loss = 0
    v_epoch +=1
    for  data in trainloader:
            
            optimizer.zero_grad() 

            # get the inputs
            list_feature, list_adj,list_bio, cas_efficiency = data
            list_feature = list_feature.cuda().float()
            list_adj = list_adj.cuda().float()
            list_bio = list_bio.cuda().float()
                     
            cas_efficiency = cas_efficiency.view(-1, 1)
            cas_efficiency =  cas_efficiency.cuda().float()
            outputs= net(list_feature,list_adj,list_bio)                   
            loss = criterion(outputs, cas_efficiency)
            train_loss += loss.item()            
            loss.backward()
            optimizer.step()

    train_loss = train_loss / len(trainloader)


    net.eval()
    
    val_loss = 0 
    with torch.no_grad():
                                 
        for data in valloader:
                
                list_feature, list_adj,list_bio, cas_efficiency = data
                list_feature = list_feature.cuda().float()
                list_adj = list_adj.cuda().float()
                list_bio = list_bio.cuda().float()
                cas_efficiency = cas_efficiency.view(-1, 1)
                cas_efficiency =  cas_efficiency.float().cuda()  
                outputs= net(list_feature,list_adj,list_bio)
                loss = criterion(outputs, cas_efficiency)
                val_loss += loss.item()

        val_loss = val_loss / len(valloader)
        val_min.append(val_loss)

        if min_loss > min(val_min):
            min_loss = val_loss
            torch.save(net.state_dict(),'/home/jovyan/Minsu/sgRNA/sgRNA/transcrip.pt')




    
    print('Epoch {}, Loss(train/val) {:2.4f}/{:2.4f}'.format(v_epoch, train_loss, val_loss))





with torch.no_grad():
        
        net = transripr(44,64,[500,250,125,1])
        print(min_loss)
        net.load_state_dict(torch.load('transcrip.pt'))
        net.cuda()
        efficiency_total = list()
        pred_efficiency_total = list()

        for data in testloader:

            list_feature, list_adj,list_bio, cas_efficiency = data
            list_feature = list_feature.cuda().float()
            list_adj = list_adj.cuda().float()
            list_bio = list_bio.cuda().float()        
            cas_efficiency = cas_efficiency.view(-1, 1)
            cas_efficiency =  cas_efficiency.float().cuda()  
            outputs= net(list_feature,list_adj,list_bio)        
            efficiency_total += cas_efficiency.tolist()
            cas_efficiency = cas_efficiency.view(-1, 1)
            pred_efficiency_total += outputs.view(-1).tolist()

        
        Spear = stats.spearmanr(efficiency_total, pred_efficiency_total)
        mse = mean_squared_error(efficiency_total, pred_efficiency_total)
     
            
        print("Transcripr: ",Spear)
        print("Transcripr: ",mse)


epoch = 1


from lstm import Lstmfc

net = Lstmfc(44,80,1)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.0005)
net.cuda()
val_min = []
min_loss = +10000
v_epoch = 0


for i in tqdm(range(epoch)):
    net.train()
    train_loss = 0
    v_epoch +=1
    
    for  data in trainloader:
            
            optimizer.zero_grad() 

            # get the inputs
            list_feature, list_adj,list_bio, cas_efficiency = data
            list_feature = list_feature.cuda().float()
            list_adj = list_adj.cuda().float()
            list_bio = list_bio.cuda().float()            
            cas_efficiency = cas_efficiency.view(-1, 1)
            cas_efficiency =  cas_efficiency.cuda().float()
            outputs= net(list_feature,list_bio)
            loss = criterion(outputs, cas_efficiency)
            train_loss += loss.item()           
            loss.backward()
            optimizer.step()

    train_loss = train_loss / len(trainloader)


    net.eval()
    val_loss = 0 
    with torch.no_grad():

        for data in valloader:
                
                list_feature, list_adj,list_bio, cas_efficiency = data
                list_feature = list_feature.cuda().float()
                list_adj = list_adj.cuda().float()
                list_bio = list_bio.cuda().float()  
                cas_efficiency = cas_efficiency.view(-1, 1)
                cas_efficiency =  cas_efficiency.float().cuda()  
                outputs= net(list_feature,list_bio)                        
                loss = criterion(outputs, cas_efficiency)
                val_loss += loss.item()

        val_loss = val_loss / len(valloader)
        val_min.append(val_loss)

        if min_loss > min(val_min):
            min_loss = val_loss
            torch.save(net.state_dict(),'/home/jovyan/Minsu/sgRNA/sgRNA/lstm.pt')



    
    print('Epoch {}, Loss(train/val) {:2.4f}/{:2.4f}'.format(v_epoch, train_loss, val_loss))





    with torch.no_grad():
            net = Lstmfc(44,80,1)
            net.load_state_dict(torch.load('lstm.pt'))
            net.cuda()
            efficiency_total = list()
            pred_efficiency_total = list()

            for data in testloader:

                list_feature, list_adj,list_bio, cas_efficiency = data
                list_feature = list_feature.cuda().float()
                list_adj = list_adj.cuda().float()
                list_bio = list_bio.cuda().float()   

                cas_efficiency = cas_efficiency.view(-1, 1)
                cas_efficiency =  cas_efficiency.float().cuda()  
                outputs= net(list_feature,list_bio)
                efficiency_total += cas_efficiency.tolist()
                cas_efficiency = cas_efficiency.view(-1, 1)   
                pred_efficiency_total += outputs.view(-1).tolist()

            
        
            Spear = stats.spearmanr(efficiency_total, pred_efficiency_total)
            mse = mean_squared_error(efficiency_total, pred_efficiency_total)
        
                
            print("LSTM: ",Spear)
            print("LSTM: ",mse)

from Mlp import MLP

net = MLP(924,[600,300,200],1)
optimizer = torch.optim.Adamax(net.parameters())
val_min = []
min_loss = +10000
v_epoch = 0
net.cuda()

for i in tqdm(range(epoch)):
    net.train()
    train_loss = 0
    v_epoch +=1
    for  data in trainloader:
            
            optimizer.zero_grad() 

            # get the inputs
            list_feature, list_adj,list_bio, cas_efficiency = data
            list_feature = list_feature.cuda().float()
            list_adj = list_adj.cuda().float()
            list_bio = list_bio.cuda().float()
            cas_efficiency = cas_efficiency.view(-1, 1)
            cas_efficiency =  cas_efficiency.cuda().float()
            outputs= net(list_feature,list_bio)
            loss = criterion(outputs, cas_efficiency)
            train_loss += loss.item()     
            loss.backward()
            optimizer.step()

    train_loss = train_loss / len(trainloader)


    net.eval()
    
    val_loss = 0 
    with torch.no_grad():
                                 
        for data in valloader:
                
                list_feature, list_adj,list_bio, cas_efficiency = data
                list_feature = list_feature.cuda().float()
                list_adj = list_adj.cuda().float()
                list_bio = list_bio.cuda().float()
                cas_efficiency = cas_efficiency.view(-1, 1)
                cas_efficiency =  cas_efficiency.float().cuda()  
                outputs= net(list_feature,list_bio)                        
                loss = criterion(outputs, cas_efficiency)
                val_loss += loss.item()

        val_loss = val_loss / len(valloader)
        val_min.append(val_loss)

        if min_loss > min(val_min):
            min_loss = val_loss
            torch.save(net.state_dict(),'/home/jovyan/Minsu/sgRNA/sgRNA/Mlp.pt')




    
    print('Epoch {}, Loss(train/val) {:2.4f}/{:2.4f}'.format(v_epoch, train_loss, val_loss))





with torch.no_grad():
        net = MLP(924,[600,300,200],1)
        print(min_loss)
        net.load_state_dict(torch.load('Mlp.pt'))
        net.cuda()
        efficiency_total = list()
        pred_efficiency_total = list()
        for data in testloader:

            list_feature, list_adj,list_bio, cas_efficiency = data
            list_feature = list_feature.cuda().float()
            list_adj = list_adj.cuda().float()
            list_bio = list_bio.cuda().float()

            cas_efficiency = cas_efficiency.view(-1, 1)
            cas_efficiency =  cas_efficiency.float().cuda()  
            outputs= net(list_feature,list_bio)
            efficiency_total += cas_efficiency.tolist()
            cas_efficiency = cas_efficiency.view(-1, 1)            
            pred_efficiency_total += outputs.view(-1).tolist()

        
      
        Spear = stats.spearmanr(efficiency_total, pred_efficiency_total)
        mse = mean_squared_error(efficiency_total, pred_efficiency_total)
      
            
        print("MLP: ",Spear)
        print("MLP: ",mse)