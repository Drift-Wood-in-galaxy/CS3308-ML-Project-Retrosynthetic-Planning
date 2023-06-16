import numpy as np
import utils
from utils import Extract_smiles as E_smiles
from utils import Extract_products as E_product
from utils import Mole_to_vec as Movec

train_file = '../data/raw_train.csv'
test_file = '../data/raw_test.csv'
validation_file = '../data/raw_val.csv'

def Pre_data(file):
    
    #data_label = np.loadtxt(fname = file,
    #                         dtype = str,
    #                         usecols = 0,
    #                         delimiter = ',',
    #                         skiprows = 1)
    
    Data = np.loadtxt(fname = file,
                               dtype = str,
                               usecols = 2,
                               delimiter = ',',
                               skiprows = 1,
                               comments = '\n')
    
    Label = []
    Feature = []
    tmp = []
 
    for data in Data:
        reactant,features = data.split('>>')
        label = E_smiles(data)

        
        tmp = features.split('.')
        if label != None:
            for feature in tmp:
                feature = Movec(str(feature))
                Feature.append(feature)
                Label.append(label)
        
        
    label_name = '../data/' + file + '_label.npy'
    np.save(label_name,Label)
    feature_name = '../data/' + file + '_data.npy'
    np.save(feature_name,Feature)

Pre_data(train_file)
Pre_data(test_file)
Pre_data(validation_file)