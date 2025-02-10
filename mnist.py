
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import matplotlib.pyplot as plt
import numpy as np
import mlp
import pickle, gzip

# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz','rb')
tset, vset, teset = pickle.load(f, encoding='bytes')
f.close()

# Just use the first few images
nread = 200

train_in = tset[0][:nread,:]

# This is a little bit of work -- 1 of N encoding
# Make sure you understand how it does it
train_tgt = np.zeros((nread,10))
print (train_tgt)
for i in range(nread):
    train_tgt[i,tset[1][i]] = 1
print (train_tgt.size)

test_in = teset[0][:nread,:]
test_tgt = np.zeros((nread,10))
for i in range(nread):
    test_tgt[i,teset[1][i]] = 1

# We will need the validation set
valid_in = vset[0][:nread,:]
valid_tgt = np.zeros((nread,10))
for i in range(nread):
    print(i, vset[1][i])
    valid_tgt[i,vset[1][i]] = 1

#train and test neural networks with different number of hidden neurons (i)
# for i in [1,2,5,10,20]:
#     print("----- "+str(i))
#     net = mlp.mlp(train_in,train_tgt,i,outtype='softmax')
#     net.earlystopping(train_in,train_tgt,valid_in,valid_tgt,0.1)
#     net.confmat(test_in,test_tgt)

net = mlp.mlp(train_in,train_tgt,10,outtype='softmax')

net.mlptrain(train_in,train_tgt,0.1,5000)
#net.earlystopping(train_in,train_tgt,valid_in,valid_tgt,0.1)

net.confmat(test_in,test_tgt)
