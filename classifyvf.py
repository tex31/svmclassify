#TEST PYSTRUCT

import itertools

import numpy as np
from scipy import sparse

from sklearn.metrics import hamming_loss
from sklearn.datasets import fetch_mldata
from sklearn.metrics import mutual_info_score
from sklearn.metrics import f1_score
from scipy.sparse.csgraph import minimum_spanning_tree

from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelClf
from pystruct.datasets import load_scene


def check_exactmatchratio(Ygt, Ypredict,datalen):
    exact = 0.0
    for t in range(datalen):
        #print Ygt[t,:]
        #print Ypredict[t,:]
        gts = Ygt
        ests = Ypredict
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            if(gt==est).all():
                exact +=1
    return exact / (datalen)



#independent Model
independent_model = MultiLabelClf(inference_method='unary')
independent_ssvm = OneSlackSSVM(independent_model, C=.1, tol=0.01)

print("fitting independent model...")
independent_ssvm.fit(Xtrain, Ytrain)

#print np.vstack(independent_ssvm.predict(Xval))[1,:]

print("Test exact matching ratio: %f"
      %check_exactmatchratio(Yval, np.vstack(independent_ssvm.predict(Xval)), datatotest))

print (f1_score(Yval[3,:],np.vstack(independent_ssvm.predict(Xtrain))[3,:], average='macro'))

'''
print("Training loss independent model: %f"
      % hamming_loss(Ytrain, np.vstack(independent_ssvm.predict(Xtrain))))
print("Test loss independent model: %f"
      % hamming_loss(Yval, np.vstack(independent_ssvm.predict(Xval))))
print Yval.shape
print Yval[1,:]
      
'''
