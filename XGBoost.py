# -*- coding: utf-8 -*-
"""

Divya Machenahalli Lokesh, DXM190018
Nidhin Anisham, NXA190000
    
"""

import pandas as pd
import numpy as np
import pickle

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

def getSplitCriterions(data):
    """
    Computes all the possibile attribute value pairs in the matrix
    Returns list of tuples with (attribute,value)
    """
    
    mean = np.mean(data,axis = 0)
    criteria = []
    for i in range(np.size(data,1)):
        column = np.sort(data[:,i])
        for j in range(1,np.size(column)):
            mean = (column[j-1] + column[j])/2
            criteria.append((i,mean))
    return criteria

def getBestSplit(x):
    maxGain = float('-inf')
    s = getSplitCriterions(x[:,:-2])
    reg = 1
    criteria = None
    left = None
    right = None
    for i in s:      
        leftCondition = np.where(x[:,i[0]]<i[1])
        rightCondition = np.where(x[:,i[0]]>=i[1])
        
        leftRes = np.sum(x[:,-1][leftCondition])
        rightRes = np.sum(x[:,-1][rightCondition])
        leftProb = 0
        rightProb = 0
        
        for j in x[:,-2][leftCondition]:
            leftProb += j*(1-j)
            
        for j in x[:,-2][rightCondition]:
            rightProb += j*(1-j)
        
        leftSim = leftRes*leftRes/(leftProb+reg)
        rightSim = rightRes*rightRes/(rightProb+reg)
        gain = leftSim + rightSim
        if(gain>=maxGain):
            maxGain = gain
            right = x[rightCondition]
            left = x[leftCondition]
            criteria = i 
    return criteria,left,right
            
def getOutput(x):
    n = 0
    d = 0
    for i in x:
        n += i[-1]
        d += i[-2]*(1-i[-2])
    return n/(d+1)
    
def getTree(x,d):
    if d==0:
        return Node(getOutput(x))
    elif np.size(x,0) < 2:
        return Node(getOutput(x))
    
    criteria,left,right = getBestSplit(x)
    root = Node(criteria)
    root.left = getTree(left,d-1)
    root.right = getTree(right,d-1)
    return root
        
def XGBoostTree(x, y, depth=0, max_depth=5,iterations=50,learningRate=0.3):
    
    unique_y, count_y = np.unique(y, return_counts=True)
    p = count_y[1]/sum(count_y)
    predictions = []
    for i in y:
        if i==1:
            predictions.append(p)
        else:
            predictions.append(1-p)
    h = []
    previousProb = y
    xgdata = pd.DataFrame()
    for i in range(iterations): 
        residuals = []
        
        for j in range(y.size):
            residuals.append(y[j] - predictions[j])
        
        xgdata['p'+str(i)] = predictions
        xgdata['r'+str(i)] = residuals
        x = np.column_stack((x,np.array(predictions)))
        x = np.column_stack((x,np.array(residuals))) 
        tree = getTree(x,max_depth)  
        previousProb = x[:,-2]
        x = x[:,:-2]
        predictions = []
        for j in range(np.size(x,0)):
            
            out = getTreeValue(tree,x[j])
            prev = previousProb[j]
            p = learningRate*out
            if prev!=1 and prev!=0:
                p += np.log2(prev/(1-prev))
            
            predictions.append(np.exp(p)/(1+np.exp(p)))
        h.append(tree)  
        
        if (np.sum(np.absolute(np.array(residuals)))) < 0.3*np.size(x,0):
             break
        
    return h,xgdata

def getTreeValue(node,X):
    value = node.data
    if type(value) is not tuple:       
        return value
    else:
        if X[value[0]]>value[1]:
            return getTreeValue(node.right,X)
        else:
            return getTreeValue(node.left,X)
        
def predict(tree,X):
    predict = []
    for i in X:
        p = 0
        for j in tree[1:]:
            p += 0.3*getTreeValue(j,i)
        p = np.exp(p)/(1+np.exp(p))
        predict.append(p)   
    return predict

def getAccuracy(yPred,yTrue):
    correct = 0
    for i in range(len(yTrue)):
        if (yPred[i] > 0.5) == yTrue[i]:
            correct += 1
    return correct/len(yTrue)
    
def predictXGBoost(tree,X):
    predict = []
    for j in tree:
        p = 0
        for k in j:
            p += 0.3*getTreeValue(k,X)
        p = np.exp(p)/(1+np.exp(p))
        predict.append(p)
    return predict
        

if __name__ == "__main__":    
    print("Training XGBoost Model...")    
    data = pd.read_csv("LabelledMusicData.csv")
    
    y = data["Cluster"]
    k = len(np.unique(y))
    data = pd.get_dummies(data,columns = ["Cluster"])
    M = data.to_numpy()
    
    split = round(0.8*np.size(M,0))
    
    X_train = M[:split,:-k]
    X_test = M[split:,:-k]
    Y_train = M[:split,-k:]
    Y_test = M[split:,-k:]
    
    pred = []
    
    clusterTrees = []
    for i in range(k):
        tree,xgdata = XGBoostTree(X_train, Y_train[:,i], max_depth=5)
        clusterTrees.append(tree)
        Y_pred = predict(tree,X_test)
        pred.append(Y_pred)
        print("Accuracy of class "+str(i)+" : "+str(getAccuracy(Y_pred,Y_test[:,i])))
    
    with open("xgboost_model.dat", "wb") as f:
            pickle.dump(clusterTrees, f)
    
    print("XGBoost model created and stored in 'xgboost_model.dat'")    