#criterian
##RMSE LOSS
import model
import data
from data import*
from data import NumbersDataset, train_test
from model import CNNcom
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
###########################################################
ligand_path = "C:/Users/User/python/kiba/ligands_can.txt"
protein_path = "C:/Users/User/python/kiba/protein1.txt"
affinity_path = "C:/Users/User/python/kiba/Y"
###########################################################

class rmsloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
    def forward(self,yhat,y):
       return torch.sqrt(self.mse(yhat,y))

def train_d(train):
#criterion = nn.MSELoss()
    criterion = rmsloss()
    model2=CNNcom()
    optimizer=optim.Adam(model2.parameters(),lr=0.003)

    num_epochs = 3
    for epoch in range(num_epochs):
       r_loss = 0
       loss1 = []
       out1 = []
       count = 0
       for i, j in (train):
           m=i[0]
           m = m.reshape((4, 62, 50))
           p=i[1]
           p = p.reshape((4, 25, 600))
           output = model2(m, p)
           out1.append(output)
           loss = criterion(output, j)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           loss1.append(loss)
           count += 1
           if count == 50:
               break
    return loss1,out1



# if __name__ == '__main__':
#     dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
#     train_data,test_data=train_test(dataset,0.2).train_test_s()
#     print(len(train_data))
#     print(len(test_data))
#     traind = train_d(train_data)
#     print(traind[0])
#     print(traind[1])
