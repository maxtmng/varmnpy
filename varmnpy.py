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


#Hyperparameter Selection - return log posterior
def hyperparameter_select(data, data_np, p, T, mn_lambda):
    n = data_np.shape[1]
    d = n + 2
    k = n * (p+1)

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
                omega[lag*n+i,lag*n+i] = 1
            if lag>0:
                omega[lag*n+i,lag*n+i] = mn_lambda**2 / psi_big[i,i] / lag**2
    omega[0,0] = 10**6
        
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
    
    log_p_a = np.log(np.linalg.det(omega))*(-n/2) 
    log_p_b = np.log(np.linalg.det(x.T @ x + np.linalg.pinv(omega)))*(-n/2)
    log_p_c = np.log(np.linalg.det(resids_hat.T @ resids_hat + (B_hat - flat).T @ np.linalg.pinv(omega) \
                                   @ (B_hat - flat)))*(-(T-n-1)/2)
    
    log_posterior_y = log_p_a + log_p_b + log_p_c
    
    return log_posterior_y
    

def MC_Minnesota_A_0_posterior(data_np, p, T, mn_lambda, A_0, return_beta):
# Structural VAR (Sims and Zha 2006)
#return_beta == 1: return beta; == 2: return beta_hat; == 0: return log posterior of A_0

    #Minnesota
    n = data_np.shape[1]
    d = n + 2
    k = n * (p+1)
    
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
    omega[0,0] = 10**6
        
    x = lag_data(data_np,p, T)
    y = data_np[p:,:]
    
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
    resids_hat = y_big - x_big @ beta_hat
    resids_hat = resids_hat.reshape((T,n), order="F")
    s_hat = resids_hat.T @ resids_hat + (B_hat - flat).T @ np.linalg.pinv(omega) @ (B_hat - flat)
    
    log_p_A_0 = np.log(np.linalg.det(A_0))*(T-p) + -1/2 * np.trace(s_hat @ A_0.T @ A_0)
    
    #Draw beta
    if return_beta == 1:
        beta_sigma = np.kron(np.linalg.pinv(A_0) @ np.linalg.pinv(A_0).T, np.linalg.pinv(x.T @ x + np.linalg.pinv(omega)))
        beta = np.random.multivariate_normal(np.ndarray.flatten(beta_hat), beta_sigma)
        return beta
    elif return_beta == 2:
        return beta_hat
    elif return_beta == 3:
        sigma = np.linalg.pinv(A_0) @ np.linalg.pinv(A_0).T
        beta_sigma = np.kron(np.linalg.pinv(A_0) @ np.linalg.pinv(A_0).T, np.linalg.pinv(x.T @ x + np.linalg.pinv(omega)))
        beta = np.random.multivariate_normal(np.ndarray.flatten(beta_hat), beta_sigma)
        return beta, sigma
    else:
        return log_p_A_0

#Converting and flattening A_0 for function
def MC_Minnesota_A_0_posterior_flat(e, A_0_con, data_np, p, T, mn_lambda, return_beta):
    n = np.array(data_np).shape[1]
    A_0 = e_to_A_0(A_0_con, e)
    A_0_f = A_0.reshape((n, n), order="F")
    
    p_A_0 = MC_Minnesota_A_0_posterior(data_np, p, T, mn_lambda, A_0_f, return_beta)
    
    return -p_A_0

def A_0_const(A_0_con):
    big = 1e10
    n = np.array(A_0_con).shape[0]

    A_0_bnd = []
    counter = 0
    for i in range(n):
        for j in range(n):
            if A_0_con[j][i] == 1:
                A_0_bnd.append((0,None))
            elif A_0_con[j][i] > 0:
                A_0_bnd.append((None,None))
    return A_0_bnd

def e_to_A_0(A_0_con, e):
    n = np.array(A_0_con).shape[0]
    temp_A_0 = np.zeros([n,n])
    e_i=0
    for i in range(n):
        for j in range(n):
            if A_0_con[j][i]>0:
                temp_A_0[j][i] = e[e_i]
                e_i += 1
    return temp_A_0

def A_0_to_e(A_0_con, A_0):
    n = np.array(A_0_con).shape[0]
    temp_e=[]
    for i in range(n):
        for j in range(n):
            if A_0_con[j][i]>0:
                temp_e.append(A_0[j][i])
    return temp_e
