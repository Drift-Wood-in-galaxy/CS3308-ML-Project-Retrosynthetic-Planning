import numpy as np
import sys
import pickle
import torch
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.ensemble import AdaBoostRegressor as Ada
np.set_printoptions(threshold=np.inf)
train_file = np.load('../data/train.pkl',allow_pickle=True)
test_file = np.load('../data/test.pkl',allow_pickle=True)
__stderr__ = sys.stderr  
sys.stderr = open('error.txt', 'a')


def MLP_sklearn(train_data,test_data,id):
    X1 = np.unpackbits(train_data['packed_fp'],axis=1)
    Y1 = torch.squeeze((train_data['values'] * 100)).numpy().astype(int)
    X2 = np.unpackbits(test_data['packed_fp'],axis=1)
    Y2 = torch.squeeze((test_data['values'] * 100)).numpy().astype(int)

    
    #print('arrive',flush=True)
    model = MLPClassifier(solver='adam', alpha=1e-3, activation='tanh', learning_rate='invscaling',
                           hidden_layer_sizes=(256,512),verbose=True,max_iter=50)
    
    
    ('Start fitting',flush=True)
    model.fit(X1,Y1)
    print('Fitting done',flush=True)
    
    pkl_name = '../model/MLP_mole_' + str(id) + '.pkl'
    with open(pkl_name,'wb') as f:
        pickle.dump(model,f)

    
    #plt.figure()

def KNN_sklearn(train_data,test_data,id):
    X1 = np.unpackbits(train_data['packed_fp'],axis=1)
    Y1 = torch.squeeze((train_data['values'])).numpy()
    X2 = np.unpackbits(test_data['packed_fp'],axis=1)
    Y2 = torch.squeeze((test_data['values'])).numpy()

    
    #print('arrive',flush=True)
    model = KNN(5)
    
    print('Start fitting',flush=True)
    model.fit(X1,Y1)
    print('Fitting done',flush=True)
    
    pkl_name = '../model/KNN_mole_' + str(id) + '.pkl'
    with open(pkl_name,'wb') as f:
        pickle.dump(model,f)
        
def Ada_sklearn(train_data,test_data,id):
    X1 = np.unpackbits(train_data['packed_fp'],axis=1)
    Y1 = torch.squeeze((train_data['values'])).numpy()
    X2 = np.unpackbits(test_data['packed_fp'],axis=1)
    Y2 = torch.squeeze((test_data['values'])).numpy()

    
    #print('arrive',flush=True)
    model = Ada(DT(max_Depth=5),n_estimators=100,random_state=rng)
    
    print('Start fitting',flush=True)
    model.fit(X1,Y1)
    print('Fitting done',flush=True)
    
    pkl_name = '../model/Ada_mole_' + str(id) + '.pkl'
    with open(pkl_name,'wb') as f:
        pickle.dump(model,f)


if __name__ == '__main__':
    #print('hello',flush=True)
    MLP_sklearn(train_file,test_file,1)
    KNN_sklearn(train_file,test_file,1)
    Ada_sklearn(train_file,test_file,1)