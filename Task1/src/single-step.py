import numpy as np
import pickle
import utils
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
    
    
    template = {keys[i]:i for i in range(len(keys))}
    return template
    
def dict_increace(temp,labels):
    keys = []
    for label in labels:
        if keys.count(label) == 0:
            keys.append(label)
    
    
    for i in range(len(keys)):
        if (keys[i] in temp) == False:
            temp[keys[i]] = len(temp)
        
    return temp
    
def label_change(Labels):
    labels = [Template[Label] for Label in Labels]
    return labels

def MLP_sklearn(X1,Y1,X2,Y2,id):
    d = []
    Solver = 'adam'
    Alpha = 1e-3
    Activation = 'tanh'
    Lr = 'invscaling'
    Hidden = (256,512)
    Batch = 500
    
    model = MLPClassifier(solver=Solver, alpha=Alpha, activation=Activation, learning_rate=Lr,
                           hidden_layer_sizes=Hidden,verbose=True,max_iter=50,batch_size=Batch,learning_rate_init = 0.001)
    
    print('Model: Mlp\n')
    print('Parameters:\nsolver = ' + Solver )
    print( 'alpha = ' + str(Alpha) )
    print( 'activation = ' + Activation )
    print( 'learning_rate = ' + Lr ) 
    print( 'hidden_layer_sizes = ' + str(Hidden) )
    print( 'batch_size = ' + str(Batch) )

    Y1 = label_change(Y1)
    
    model.fit(X1,Y1)
    
    pkl_name = '../model/MLP_one_step_' + str(id) + '.pkl'
    with open(pkl_name,'wb') as f:
        pickle.dump(model,f)
    
    dict_increace(Template,test_label)
    tem_name = 'Template_one_step_MLP_' + str(id)
    np.save(tem_name,Template)
    
    Y2 = label_change(test_label)
    
    score = model.score(X2,Y2)
    d.append(score)
    
    print(score)
    
    #plt.figure()
    
def SVM_sklearn(X1,Y1,X2,Y2,id):
    c = 1.0
    Kernel = 'rbf'
    
    model = SVC(C = c, kernel = Kernel,verbose=True,max_iter=50,probability=True)
    
    print('Model: Mlp\n')
    print('Parameters:\n')
    print( 'C = ' + str(c) )
    print( 'Kernel = ' + str(Kernel) )

    Y1 = label_change(Y1)
    
    model.fit(X1,Y1)
    
    pkl_name = '../model/SVM_one_step' + str(id) + '.pkl'
    with open(pkl_name,'wb') as f:
        pickle.dump(model,f)
    
    dict_increace(Template,test_label)
    
    dict_increace(Template,test_label)
    tem_name = 'Template_one_step_SVM_' + str(id)
    np.save(tem_name,Template)
    
    Y2 = label_change(test_label)
    
    score = model.score(X2,Y2)
    
    print(score)


if __name__ == '__main__':
    Template = dict_form(train_label)

    SVM_sklearn(train_data,train_label,test_data,test_label,2)
    #MLP_sklearn(train_data,train_label,test_data,test_label,(256,512))

