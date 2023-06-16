import numpy as np
import pickle
import utils
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import Mole_to_vec as Movec
train_data = np.load('../data/raw_train.csv_data.npy')
train_label = np.load('../data/raw_train.csv_label.npy')
test_data = np.load('../data/raw_test.csv_data.npy')
test_label = np.load('../data/raw_test.csv_label.npy')

Template = {}

def dict_form(labels):
    keys = []
    for label in labels:
        if keys.count(label) == 0:
            keys.append(label)
    
    
    Template = {keys[i]:i for i in range(len(keys))}
    return Template
    
def label_change(Labels):
    labels = [Template[Label] for Label in Labels]
    return labels

def Test(X2,Y2,K):
    
    pkl_name = '../model/MLP_one_step_512.pkl'
    with open(pkl_name,'rb') as f:
        pickle_model = pickle.load(f)
        
    correct = 0
    number = len(Y2)
    i = 0
    Predict = pickle_model.predict_proba(X2)
    for p in Predict:
        pre_label = 0
        for j in range(K):
            pre_label = np.argmax(p)
            if pre_label == Y2[i]:
                correct +=1
                break
            else:
                p[pre_label] = 0
        i += 1
    
    score = correct / number
    print('K: ' + str(K))
    print('Accuracy: ' + str(score))
    


if __name__ == '__main__':
    Template = dict_form(np.hstack((train_label,test_label)))
    for k in [1,2,5,10,20]:
        Test(test_data,label_change(test_label),k)