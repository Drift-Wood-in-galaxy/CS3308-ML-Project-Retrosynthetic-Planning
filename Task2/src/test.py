import numpy as np
import math
import pickle
import sys
import torch
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
np.set_printoptions(threshold=np.inf)
train_file = np.load('../data/train.pkl',allow_pickle=True)
test_file = np.load('../data/test.pkl',allow_pickle=True)
#__stderr__ = sys.stderr  
#sys.stderr = open('error.txt', 'a')

    


def Test_MLP(test_data):
    X2 = np.unpackbits(test_data['packed_fp'],axis=1)
    Y2 = torch.squeeze((test_data['values'])).numpy()
    
    pkl_name = '../model/MLP_mole_1.pkl'
    with open(pkl_name,'rb') as f:
        pickle_model = pickle.load(f)
    
    Y3 = (pickle_model.predict(X2)).astype(float) / 100
    #print('The predict is:')
    #print(Y3)
    print('The L2 loss is:')
    print(math.sqrt(sum(np.square(Y3 - Y2))))
    print('The average of Lable is:')
    print(sum( Y2)/ len(Y2))
    print('The average loss is:')
    print(sum(abs(Y3 - Y2))/ len(Y2))

def Test(test_data):
    X2 = np.unpackbits(test_data['packed_fp'],axis=1)
    Y2 = torch.squeeze((test_data['values'])).numpy()
    
    pkl_name = '../model/KNN_mole_1.pkl'
    with open(pkl_name,'rb') as f:
        pickle_model = pickle.load(f)
    
    Y3 = (pickle_model.predict(X2))
    #print('The predict is:')
    #print(Y3)
    print('The L2 loss is:')
    print(math.sqrt(sum(np.square(Y3 - Y2))))
    print('The average of Lable is:')
    print(sum( Y2)/ len(Y2))
    print('The average loss is:')
    print(sum(abs(Y3 - Y2))/ len(Y2))
    
    
    

if __name__ == '__main__':
    Test(test_file)