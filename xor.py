

import numpy as np
import mlp
xordata = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])


q = mlp.mlp(xordata[:,0:2],xordata[:,2:3],2,outtype='logistic')
# q.mlptrain(xordata[:,0:2],xordata[:,2:3],0.25,1001)
# Early stooping succeeds 99% at 800-1000 iterations
q.earlystopping(xordata[:,0:2],xordata[:,2:3],xordata[:,0:2],xordata[:,2:3],0.25)
q.confmat(xordata[:,0:2],xordata[:,2:3])