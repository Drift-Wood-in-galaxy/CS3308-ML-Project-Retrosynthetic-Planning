import numpy as np
import torch
np.set_printoptions(threshold=np.inf)

train_data = np.load('train.pkl',allow_pickle=True)
test_data = np.load('test.pkl',allow_pickle=True)
X1 = np.unpackbits(train_data['packed_fp'],axis=1)
Y1 = torch.squeeze((train_data['values'] * 100)).numpy().astype(int)
X2 = np.unpackbits(test_data['packed_fp'],axis=1)
Y2 = torch.squeeze((test_data['values'] * 100)).numpy().astype(int)


print(type(X1))
print(type(Y1))
print(type(X2))
print(type(Y2))

print(Y2)

#print(X1.shape)
#print(Y1.shape)
#print(Y1[0:4095])
#print(X2.shape)
#print(Y2.shape)