
import matplotlib.pyplot as plt
import numpy as np
import mlp
import pickle
import gzip

# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz', 'rb')
tset, vset, teset = pickle.load(f, encoding='bytes')
f.close()


# Just use the first few images
nread = 200
train_in = tset[0][:nread, :]

#print(np.reshape(tset[0][1, :], [28, 28]))
# lets plot the first digit in the training set
plt.imshow(np.reshape(tset[0][1, :], [28, 28]))
plt.show()


