# IOE 511/MATH 562, University of Michigan
# Code written by: Jiale Zha

import numpy as np

# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1) Linear Least Squares
# (2) Logistic Regression


def least_square_func(w, X, y):
    # Function that computes the function value for the Linear Least Squares function
    #
    #           Input: w, X, y
    #           Output: F(w) = sum_{i=1}^n (X_iw - y_i)_2^2/(2n) = ||Xw - y||^2_2/(2n)
    #
    return np.mean((X@w - y)**2)/2



def least_square_grad(w, X, y):
    # Function that computes the gradient of the Linear Least Squares function
    #
    #           Input: w, X, y
    #           Output: g = nabla F(w)
    #
    return (X.T@X@w - (X.T@y).reshape(-1))/len(y)

def least_square_pred(w, X):
    # Function that computes the prediction from the Linear Least Squares function
    #
    #           Input: w, X, y
    #           Output: prediction = sign(Xw)
    #
    return np.sign(X@w) 



def logistic_func(w, X, y):
    # Function that computes the function value for the Logistic Regression function
    #
    #           Input: w, X, y
    #           Output: F(w) = \sum_{i = 1}^n log(1 + exp(-y_iX_iw))/n
    
    return np.mean(np.log(1 + np.exp(-y * (X@w))))

def logistic_grad(w, X, y):
    # Function that computes the gradient of the Logistic Regression function
    #
    #           Input: w, X, y
    #           Output: g = nabla F(w)
    #                   where F(w) = \sum_{i = 1}^n log(1 + exp(-y_iX_iw))/n

    # temp = np.exp(y * (X@w))
    # return np.mean((-y / (1 + temp)).reshape((-1, 1)) * X, axis=0)
    temp = np.exp(y * (-X@w))
    
    return np.mean((-y * temp / (1 + temp)).reshape((-1, 1)) * X, axis=0)

def logistic_pred(w, X):
    return np.sign(X@w) 
    
