import numpy as np

#np.set_printoptions(threshold=np.inf)

X1= np.load('raw_train.csv_data.npy')
Y1 = np.load('raw_train.csv_label.npy')
X2 = np.load('raw_test.csv_data.npy')
Y2= np.load('raw_test.csv_label.npy')

print(type(X1))
print(type(Y1))
print(type(X2))
print(type(Y2))

print(X1.shape)
print(Y1.shape)
print(Y1[0:4095])
print(X2.shape)
print(Y2.shape)