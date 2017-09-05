#get image feature and label
import sys
import numpy as np
import os
import logging
import re
import string
import bitarray
import time

tfmrnn_root = '/home/nnyhoavy/tfmrnn/TF-mRNN-master/'
#tfmrnn_root = '/home/misa/TF-mRNN-master/'  # this file is expected to be in {caffe_root}/examples
#/home/misa-master2/tf-mrnn/TF-mRNN
sys.path.append(tfmrnn_root + 'py_lib')

#from common_utils import CommonUtiler
 #pour extraire les annotations
#%matplotlib inline
from pycocotools.coco import COCO

vf_dir = tfmrnn_root+'cache/mscoco_image_features/inception_v3'
#/home/misa-master2/tf-mrnn/TF-mRNN/cache/mscoco_image_features/inception_v3

'''print ("...........coco initialization")
   #initalizer les annotations instances 
dataDir=tfmrnn_root+'datasets/ms_coco'
#/home/misa-master2/tf-mrnn/TF-mRNN/datasets/ms_coco
dataType='train2014'
annotationType = 'instances'
annFile='%s/annotations%s/%s_%s.json'%(dataDir,annotationType,annotationType,dataType)
print("...........coco initialized")
coco=COCO(annFile)'''
    
def load_data(anno_file_path, coco, vf_dir, datalength) :
    num_failed = 0
    datafinale = []
    #data = {}
    categorie = 91*[0]
    annos = np.load(anno_file_path).tolist()
    #features= np.zeros(shape=(50000,2086))
    #label = np.zeros(shape=(50000,91))
    features= []
    label = []
    itt =0
    for (ind_a, anno) in enumerate(annos):
            #print ("annotation........",anno['id'])
            #load annoation instance
            #img = coco.loadImgs(anno['id'])
            annIds = coco.getAnnIds(imgIds=anno['id'])
            anns = coco.loadAnns(annIds)
            catsIds = []
            for ann in anns:
                catsIds.append(ann['category_id'])
            catsIds = sorted(list(set(catsIds)))
            #data['categorie']=catsIds
            for cat in catsIds:
                #print 'cat....:', cat
                categorie[cat]=1 
            # Load visual features
            feat_path = os.path.join(vf_dir, anno['file_path'],
            anno['file_name'].split('.')[0] + '.txt')
            #print 'feat_path ',feat_path 
            if os.path.exists(feat_path):
                vf = np.loadtxt(feat_path)
            else:
                num_failed += 1
                continue
            #data['visual_features'] = vf
            #print 'data[categorie]', data['categorie']
            #print 'data[vf]', vf[2]
            #datafinale.append((vf,categorie))
            #features.data[itt, ...] = vf
            #label.data[itt, ...]= categorie
            features.append(vf)
            label.append(categorie)
            itt = itt +1
            if(itt == datalength):
                print ("tapitraaa")
                break
            
            
    print ('data finale  ',len(datafinale),' num_failed', num_failed)
    return features, label

datatotrain = 50000
datatotest = 10000000
#############TRAIN###################
anno_files_path_train = tfmrnn_root+"datasets/ms_coco/mscoco_anno_files/anno_list_mscoco_train_m_RNN.npy"    

print ("...........coco initialization")
   #initalizer les annotations instances 
dataDir=tfmrnn_root+'datasets/ms_coco'
#/home/misa-master2/tf-mrnn/TF-mRNN/
dataType='train2014'
annotationType = 'instances'
annFile='%s/annotations%s/%s_%s.json'%(dataDir,annotationType,annotationType,dataType)
print("...........coco initialized")
coco=COCO(annFile)

a,b= load_data(anno_files_path_train, coco, vf_dir, datatotrain)
Xtrain = np.array(a)
Ytrain = np.array(b)
del coco

###################VAL#############
print ("...........coco initialization")
   #initalizer les annotations instances 
dataDir=tfmrnn_root+'datasets/ms_coco'
#/home/misa-master2/tf-mrnn/TF-mRNN/
dataType='val2014'
annotationType = 'instances'
annFile='%s/annotations%s/%s_%s.json'%(dataDir,annotationType,annotationType,dataType)
print ("...........coco initialized")
coco=COCO(annFile)

anno_files_path_val= tfmrnn_root+"datasets/ms_coco/mscoco_anno_files/anno_list_mscoco_modelVal_m_RNN.npy"    

a,b = load_data(anno_files_path_val, coco, vf_dir, datatotest)
Xval = np.array(a)
Yval = np.array(b)

np.savetxt("Xtrain.txt",Xtrain)
np.savetxt("Ytrain.txt",Ytrain)
np.savetxt("Xval.txt",Xval)
np.savetxt("Yval.txt",Yval)

print("ok")
