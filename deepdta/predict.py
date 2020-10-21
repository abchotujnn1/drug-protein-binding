import data
from data import*
#import model
#from model import*
import train
from train import*
#from train import
##############################################
ligand_path = "C:/Users/User/python/kiba/ligands_can.txt"
protein_path = "C:/Users/User/python/kiba/protein1.txt"
affinity_path = "C:/Users/User/python/kiba/Y"
##############################################
def predict(test):
    out2 = []
    count = 0
    for i,j in(test):
        m1 = i[0]
        m1 = m1.reshape((1, 62, 50))
        p1 = i[1]
        p1 = p1.reshape((1, 25, 600))
        predict = model2(m1, p1)
        out2.append(predict)
        count += 1
        if count == 50:
            break
    return out2
if __name__=='__main__':
      dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
      train_data,test_data=train_test(dataset,0.4).train_test_s()
      print(len(train_data))
      print(len(test_data))
      pr=predict(test_data)
      print(pr)