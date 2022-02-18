#!/usr/bin/env python
# coding: utf-8

# Vector Autoregressive (VAR) Models with Minnesota prior (Doan et al., 1984) implementation on Python 

# $y_t = c+B_1 y_{t_1} + ⋯ + B_p y_{t-p}+ \varepsilon_t$
# ,where $\varepsilon_t \sim N(0,\Sigma)$
# 
# In vector form,
# $Y = X \beta + e$
# ,where $e \sim N(0,\Sigma \bigotimes I_T)$



import numpy as np 
import pandas as pd
import math
from scipy.stats import invwishart
from scipy.stats import multivariate_normal



#Processing and returning data of period T after a lag of p
def lag_data(data_np, p, T):
    n = data_np.shape[1]
    k = n * (p+1)
    x_small = np.zeros([T, k])
    for i in range(p+1):
        if i==0:
            x_small[:, i*n:(i+1)*n] = np.ones([T,n])
        else:
            x_small[:, i*n:(i+1)*n] = data_np[(p-i):(T+p-i), :n]
    return x_small


# In[1]:


def MC_Minnesota_prior(data_np, p, mn_lambda):
    T = data_np.shape[0] - p
    n = data_np.shape[1]
    k = n * (p+1)
    
    #Minnesota
    d = n + 2

    psi_big = np.zeros([n,n])

    #AR(1) estimation for psi_big
    for i in range(n):
        ar_x = data_np[p-1:-1,i]
        ar_y = data_np[p:,i]

        A = np.vstack([ar_x, np.ones(len(ar_x))]).T
        m, c = np.linalg.lstsq(A, ar_y, rcond=None)[0]

        ar_resids = ar_y - ar_x * m - c

        psi_big[i,i] = np.var(ar_resids)
        
    
    #Omega
    omega = np.zeros([k,k])
    for lag in range(p+1):
        for i in range(n):
            if lag==0:
                omega[lag*n+i,lag*n+i] = 10**6
            if lag>0:
                omega[lag*n+i,lag*n+i] = mn_lambda**2 / psi_big[i,i] / lag**2
    
        
    x = lag_data(data_np,p, T)
    y = data_np[p:,:n]
    
    #Formulate vector forms
    x_big = np.kron(np.identity(n), x)
    y_big = y.reshape((-1, 1), order="F")


    beta_mean = np.zeros([k*n,1])
    for i in range(n):
        for lag in range(p+1):
            if lag==1:
                beta_mean[i*k+lag*n+i] = 1
    
    flat = beta_mean.reshape((k, n), order="F")
    
    B_hat = np.linalg.pinv(x.T @ x + np.linalg.pinv(omega)) @ (x.T @ y + np.linalg.pinv(omega) @ flat)
    beta_hat = B_hat.reshape((-1, 1), order="F")

    #Cal s_OLS
    resids_hat = y_big - np.matmul(x_big, beta_hat)
    resids_hat = resids_hat.reshape((T,n), order="F")
    s_hat = resids_hat.T @ resids_hat + psi_big + (B_hat - flat).T @ np.linalg.pinv(omega) @ (B_hat - flat)
    
    
    #Draw sigma
    sigma = invwishart.rvs(T+d, s_hat)

    #Draw beta
    beta_sigma = np.kron(sigma, np.linalg.pinv(x.T @ x + np.linalg.pinv(omega)))
    beta = np.random.multivariate_normal(np.ndarray.flatten(beta_hat), beta_sigma)

    return beta, sigma



#Forecasting based on beta and sigmma
def predict(data_for, H, p, beta, sigma):
    n = data_for.shape[1]
    if p == None:
        p = data_for.shape[0]
    data_for_np = data_for[-p:]
    
    for i in range(H):
        x = lag_data(data_for_np[i:i+p,:], p, 1)
        x_big = np.kron(np.identity(n), x)

        shock = np.random.multivariate_normal(np.zeros([n]), np.kron(sigma, np.identity(1)))
        Y_big = np.matmul(x_big, beta) + shock

        data_for_np = np.append(data_for_np, Y_big.reshape([1,n]), axis=0)

    return data_for_np[p:]

