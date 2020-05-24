# -*- coding: utf-8 -*-
"""

Divya Machenahalli Lokesh, DXM190018
Nidhin Anisham, NXA190000
    
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def getROC(Y_pred,Y):
    pred = np.array(Y_pred)
    Y_test = pd.DataFrame()
    Y_test["Cluster"] = Y
    Y_test = pd.get_dummies(Y_test,columns = ["Cluster"])
    Y_test = Y_test.to_numpy()
    
    for i in range(np.size(pred,1)):  
        y_col = np.transpose(np.vstack((pred[:,i],Y_test[:,i])))     
        y_col = y_col[y_col[:,0].argsort()[::-1]]
        unique_y, count_y = np.unique(y_col[:,1], return_counts=True)
 
        x = [0]
        y = [0]
        label = "Cluster" + str(i+1)
        prev = y_col[0][1]
        tpr = 0
        fpr = 0
        for j in range(np.size(y_col,0)):
            if(y_col[j][1]==prev):
                if(y_col[j][1] == 1):
                    tpr += 1
                else:
                    fpr += 1
            else:
                x.append(fpr/count_y[0])
                y.append(tpr/count_y[1])
                prev = y_col[j][1]
                
        x.append(fpr/count_y[0])
        y.append(tpr/count_y[1])
        plt.plot(x, y, label = label)       
        auc = roc_auc_score(Y_test[:,i], pred[:,i])
        print("AUC for Cluster "+str(i)+": %.3f" % auc)
        
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve') 
    plt.legend() 
    plt.show() 
    
       
def confusion_matrix(Y_True,Y_Pred):
    
    k=len(np.unique(Y_True))
    confusion_matrix=[[] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            count=0
            for a in range(len(Y_True)):
                if Y_Pred[a]==i and Y_True[a]==j:
                    count+=1
            confusion_matrix[i].append(count)
    return np.array(confusion_matrix)
        

def precision_recall(confusion_matrix):
    pre_re=[]
    cs=np.sum(confusion_matrix,axis=0)
    for i in range(len(confusion_matrix)):
        s=np.sum(confusion_matrix[i])
        precision=round((confusion_matrix[i][i]/s)*100,4)
        recall=round((confusion_matrix[i][i]/cs[i])*100,4)
        pre_re.append((precision,recall))
    return np.array((pre_re))  


def get_tpr_fpr(confusion_matrix):
    tpr_fpr=[]
    cs=np.sum(confusion_matrix,axis=0) 
    rw=np.sum(confusion_matrix,axis=1)
    total_sum=np.sum(confusion_matrix)
    for i in range(len(confusion_matrix)):
        tpr=round((confusion_matrix[i][i]/cs[i])*100,4)
        num=rw[i]-confusion_matrix[i][i]
        den=total_sum-cs[i]
        fpr=round((num/(num+total_sum))*100,4)
        tpr_fpr.append((tpr,fpr))
    return np.array(tpr_fpr)

    