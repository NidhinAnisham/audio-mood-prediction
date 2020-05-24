# -*- coding: utf-8 -*-
"""

Divya Machenahalli Lokesh, DXM190018
Nidhin Anisham, NXA190000
    
"""

from XGBoost import predictXGBoost
from SVM import svm_test
from getDataSet import getDataset
import numpy as np
from os import path
import pickle

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        
if __name__ == "__main__":
    
    filename = input("Enter audio file name: ")
    #filename = "Belageddu - Kirik Party Rakshit Shetty Vijay Prakash B Ajaneesh Lokanath.mp3"        
    data = getDataset([filename])[0]
    print("File Processed.")
    if path.exists("Kmeans_labels.dat"):
         with open("Kmeans_labels.dat","rb") as f:
             clusterLabels = pickle.load(f)
             
                          
    if path.exists("xgboost_model.dat"):
        with open("xgboost_model.dat","rb") as f:
            clusterTrees = pickle.load(f)
        xgBootPrediction=predictXGBoost(clusterTrees,np.array(data)[0])
        
        print('\nPrediction of Audio Mood using XGBOOST')
        s=''
        for i in range(len(xgBootPrediction)):
            confidence=round(xgBootPrediction[i]*100,2)
            s+=clusterLabels[i].capitalize()+':'+str(confidence)+'%    '
        print(s)
        
        
    else:
        print("XGBoost Model not available")
    
    if path.exists("svm_model.dat"):
        with open("svm_model.dat","rb") as f:
            svm_coeff = pickle.load(f)
        svmPrediction=np.array(svm_test(svm_coeff,np.array(data)))
        
        svmPrediction=list(svmPrediction.flatten())
        
        print('\nPrediction of Audio Mood using SVM')
        s=''
        for i in range(len(svmPrediction)):
            confidence=round(svmPrediction[i]*100,2)
            s+=clusterLabels[i].capitalize()+':'+str(confidence)+'%  '
        print(s)
    else:
        print("SVM Model not available")