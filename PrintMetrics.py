# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:20:15 2020

@author: mldiv
"""
from SVM import svm_test,svm_score
from XGBoost import predictXGBoost
from DecisionTrees import discretize,predict_example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
import pickle
from getMetrics import getROC,confusion_matrix,precision_recall,get_tpr_fpr

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

if __name__=='__main__':
     
    
     data_normalized = pd.read_csv('LabelledMusicData.csv') 
     M=data_normalized.to_numpy()
   
     row=round(0.8*np.size(M,0))
     X_test = M[row:,:-1]
     Y_test = M[row:,-1]
     
                 
     if path.exists("svm_model.dat"):
        with open("svm_model.dat","rb") as f:
            svm_coeff = pickle.load(f)

        y_pred=svm_test(svm_coeff,X_test)         
        roc_y_pred=np.transpose(y_pred)
         
        print('Plotting ROC for SVM...')
        getROC(roc_y_pred,Y_test)
         
        y_sum=np.sum(y_pred,axis=0)
        y_pred=y_pred/y_sum
        final_class=np.argmax(y_pred,axis=0)
         
        print('\nConfusion Matrix of test set for each cluster using SVM.')
        con_matrix=confusion_matrix(Y_test,final_class)
        print(con_matrix)
         
        print('\nPrecision and Recall of test set  for each cluster using SVM.')
        pre_recall=precision_recall(con_matrix)
        print(pre_recall)
         
        print('\nTrue Positive Rate and False Positive Rate of test set for each cluster using SVM')
        t_f_p=get_tpr_fpr(con_matrix)
        print(t_f_p)    
        print('\nAccuracy of SVM for given Test Data:', round(svm_score(final_class,Y_test)*100,3))
         
     else:
        print("SVM Model not available")
    
     xgBootPrediction=[]
     if path.exists("xgboost_model.dat"):
        with open("xgboost_model.dat","rb") as f:
            clusterTrees = pickle.load(f)
            
        for i in X_test:
            xgBootPrediction.append(predictXGBoost(clusterTrees,i))
         
        print('\nPlotting ROC for XGBOOST...')
        getROC(xgBootPrediction,Y_test)
        
        xgBootPrediction=np.transpose(xgBootPrediction)
        y_sum=np.sum(xgBootPrediction,axis=0)
        xgBootPrediction=xgBootPrediction/y_sum
        final_class=np.argmax(xgBootPrediction,axis=0)
         
        print('\nConfusion Matrix of test set for each cluster using XGBOOST.')
        con_matrix=confusion_matrix(Y_test,final_class)
        print(con_matrix)
         
        print('\nPrecision and Recall of test set  for each cluster using XGBOOST.')
        pre_recall=precision_recall(con_matrix)
        print(pre_recall)
         
        print('\nTrue Positive Rate and False Positive Rate of test set for each cluster using XGBOOST')
        t_f_p=get_tpr_fpr(con_matrix)
        print(t_f_p)    
        print('\nAccuracy of XGBOOST for given Test Data:', round(svm_score(final_class,Y_test)*100,3))                           
     else:
        print("XGBOOST Model not available")  
        
     