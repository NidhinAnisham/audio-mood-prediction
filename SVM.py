# -*- coding: utf-8 -*-
"""

Divya Machenahalli Lokesh, DXM190018
Nidhin Anisham, NXA190000
    
"""

import numpy as np
import pandas as pd
from os import path
import pickle


def kernel(x1, x2):
    gamma = 5.0
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-gamma * distance)

    
def fit(X, Y, C , max_iter, tol, support_vector_tol):
    # Compute coefficients of the dual problem
    n_samples, n_features = np.shape(X)
    lagrange_multipliers, intercept = svm_compute_weights(X, Y,C,max_iter,tol)
    support_vector_indices = lagrange_multipliers > support_vector_tol
    dual_coef = lagrange_multipliers[support_vector_indices] * Y[support_vector_indices] # alpha_i
    support_vectors = X[support_vector_indices]
    return support_vectors,dual_coef,intercept


def svm_compute_intercept(alpha,yg,C):
    indices = (alpha < C) * (alpha > 0)
    return np.mean(yg[indices])
    
    
def svm_compute_kernel_matrix_row(X, index):
    row = np.zeros(X.shape[0])
    x_i = X[index, :]
    for j,x_j in enumerate(X):
        row[j] = kernel(x_i, x_j)
    return row
 
   
def svm_compute_weights(X, y,C,max_iter,tol):
    # Solver to compute the intercept and lagrange multipliers
    iteration = 0
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples) # Initialise coefficients to 0  w
    g = np.ones(n_samples) # Initialise gradients to 1

    while True:
        yg = g * y
        # Working Set Selection via maximum violating constraints
        indices_y_positive = (y == 1)
        indices_y_negative = (np.ones(n_samples) - indices_y_positive).astype(bool)
        indices_alpha_upper = (alpha >= C)
        indices_alpha_lower = (alpha <= 0)

        indices_violate_Bi = (indices_y_positive * indices_alpha_upper) + (indices_y_negative * indices_alpha_lower)
        yg_i = yg.copy()
        yg_i[indices_violate_Bi] = float('-inf') #cannot select violating indices
        indices_violate_Ai = (indices_y_positive * indices_alpha_lower) + (indices_y_negative * indices_alpha_upper)
        yg_j = yg.copy()
        yg_j[indices_violate_Ai] = float('+inf') #cannot select violating indices
        
        i = np.argmax(yg_i)
        j = np.argmin(yg_j)
        
        # Stopping criterion: stationary point or maximum iterations
        stop_criterion = yg_i[i] - yg_j[j] < tol
        if stop_criterion or (iteration >= max_iter and max_iter != -1):
            break

        # Compute lambda via Newton Method and constraints projection
        lambda_max_1 = (y[i] == 1) * C - y[i] * alpha[i]
        lambda_max_2 = y[j] * alpha[j] + (y[j] == -1) * C
        lambda_max = np.min([lambda_max_1, lambda_max_2])

        Ki = svm_compute_kernel_matrix_row(X, i)
        Kj = svm_compute_kernel_matrix_row(X, j)
        lambda_plus = (yg_i[i] - yg_j[j]) / (Ki[i] + Kj[j] - 2 * Ki[j])
        lambda_param = np.max([0, np.min([lambda_max, lambda_plus])])

        # Update gradient
        g = g + lambda_param * y * (Kj - Ki)

        # Direction search update
        alpha[i] = alpha[i] + y[i] * lambda_param
        alpha[j] = alpha[j] - y[j] * lambda_param

        iteration += 1

    # Compute intercept
    intercept = svm_compute_intercept(alpha, yg,C)

    return alpha, intercept


def svm_compute_kernel_support_vectors( X,support_vectors):
    res = np.zeros((X.shape[0], support_vectors.shape[0]))
    for i,x_i in enumerate(X):
        for j,x_j in enumerate(support_vectors):
            res[i, j] = kernel(x_i, x_j)
    return res


def svm_predict(X,support_vectors,dual_coef,intercept):
    # Given a new datapoint, predict its label
    kernel_support_vectors = svm_compute_kernel_support_vectors(X,support_vectors)
    prediction = intercept + np.sum(np.multiply(kernel_support_vectors, dual_coef),1)
    prediction=np.exp(prediction)/(1+np.exp(prediction))
    return prediction
        

def svm_score( predictions, y):
    # Compute proportion of correct classifications given true labels
    scores = predictions == y
    return sum(scores) / len(scores)
 
   
def get_svm_model(X_train,Y_train,C=1.0,max_iter=1000,tol=0.001,support_vector_tol=0.01):
     
    k=len(np.unique(Y_train))
    
    coefficients=[]
    for j in range(k):
        Y=Y_train.copy()
        for i in range(len(Y_train)):
            
            if Y[i]!=j:
                Y[i]=-1
            else:
                Y[i]=1
           
        #support_vectors,dual_coef,intercept=fit(X_train,Y,C,max_iter,tol,support_vector_tol)
        coefficients.append(fit(X_train,Y,C,max_iter,tol,support_vector_tol))
       
    with open("svm_model.dat", "wb") as f:
        pickle.dump(coefficients, f)
    return coefficients


def svm_test(coefficients,X_test):
    
    y_pred=[]
    for parmeter in coefficients:
        y_pred.append(svm_predict(X_test,parmeter[0],parmeter[1],parmeter[2]))      
    return y_pred
    
if __name__=='__main__':

     data_normalized = pd.read_csv('LabelledMusicData.csv') 
     M=data_normalized.to_numpy()
   
     row = round(0.8*np.size(M,0))
     X_train = M[:row,:-1]
     X_test = M[row:,:-1]
     Y_train = M[:row,-1]
     Y_test = M[row:,-1]
     
     print('Training SVM model...')
     coefficients=get_svm_model(X_train,Y_train)
     print("SVM model created and stored in 'svm_model.dat'")