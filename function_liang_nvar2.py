 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to compute the transfer of information from variable xj to variable xi (T) and the corresponding normalization (tau)
Multivariate time series
Liang (2021, 'Normalized multivariate time series causality analysis and causal graph reconstruction')

Version designed to output only 1 bootstrap realization (launch in parallel)

Last updated: 18/10/2021

@author: David Docquier
"""

import numpy as np
import numba
from numba import jit

@jit(nopython=True)
def compute_liang_nvar(x,dt):
    
    # Function to compute absolute transfer of information from xj to xi (T)
    def compute_liang_index(detC,Deltajk,Ckdi,Cij,Cii):
        T = (1. / detC) * np.sum(Deltajk * Ckdi) * (Cij / Cii) # absolute rate of information flowing from xj to xi (nats per unit time) (equation (14))
        return T
    
    # Function to compute relative transfer of information from xj to xi (tau)
    def compute_liang_index_norm(detC,Deltaik,Ckdi,T_all,Tii,gii,Cii,Tji):
        selfcontrib = (1. / detC) * np.sum(Deltaik * Ckdi) # contribution from itself (equation (15))
        transfer = np.sum(np.abs(T_all)) - np.abs(Tii) # transfer contribution (equation (20))
        noise = 0.5 * gii / Cii # noise contribution
        Z = np.abs(selfcontrib) + transfer + np.abs(noise) # normalizer (equation (20))
        tau = 100. * Tji / Z # relative rate of information flowing from xj to xi (%) (equation (19))
        return tau
    
    # Dimensions
    nvar = x.shape[0] # number of variables
    N = x.shape[1] # length of the time series (number of observations)
    
    # Compute tendency dx
    k = 1 # k = 1 (or 2 for highly chaotic and densely sampled systems)
    dx = np.zeros((nvar,N)) # initialization of dx (to have the same number of time steps as x)
    for i in np.arange(nvar):
        dx[i,0:N-k] = (x[i,k:N] - x[i,0:N-k]) / (k * dt) # Euler forward finite difference of x (equation (7))
    
    # Compute covariances and matrix determinant
    C = np.cov(x) # covariance matrix
    dC = np.empty_like(C) * 0.
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            dC[j,i] = (np.sum((x[j,:] - np.nanmean(x[j,:])) * (dx[i,:] - np.nanmean(dx[i,:])))) / (N - 1.) # covariance between x and dx
    detC = np.linalg.det(C) # matrix determinant
    
    # Compute cofactors
    Delta = np.linalg.inv(C).T * detC # cofactor matrix (https://en.wikipedia.org/wiki/Minor_(linear_algebra))
    
    # Compute absolute transfer of information (T) and correlation coefficient
    T = np.zeros((nvar,nvar))
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            T[j,i] = compute_liang_index(detC,Delta[j,:],dC[:,i],C[i,j],C[i,i]) # compute T (transfer of information from xj to xi) and create matrix

    # Compute noise terms
    g = np.zeros(nvar)
    for i in np.arange(nvar):
        a1k = np.dot(np.linalg.inv(C),dC[:,i]) # compute a1k coefficients based on matrix-vector product (see beginning of page 4 in Liang (2014))
        f1 = np.nanmean(dx[i,:])
        for k in np.arange(nvar):
            f1 = f1 - a1k[k] * np.nanmean(x[k,:])
        R1 = dx[i,:] - f1
        for k in np.arange(nvar):
            R1 = R1 - a1k[k] * x[k,:]
        Q1 = np.sum(R1**2.)       
        g[i] = Q1 * dt / N # equation (10)
    
    # Compute relative transfer of information (tau)
    tau = np.zeros((nvar,nvar))
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            tau[j,i] = compute_liang_index_norm(detC,Delta[i,:],dC[:,i],T[:,i],T[i,i],g[i],C[i,i],T[j,i]) # compute tau and create matrix
    
    # Compute error in Tji and tauji using bootstrap with replacement
    boot_T = np.zeros((nvar,nvar))
    boot_tau = np.zeros((nvar,nvar))
    
    # Resample x and dx
    index = np.arange(N)
    boot_index = np.random.choice(index,N,replace=True)
    boot_x = np.zeros((nvar,N))
    boot_dx = np.zeros((nvar,N))
    for t in np.arange(N):
        boot_x[:,t] = x[:,boot_index[t]]
        boot_dx[:,t] = dx[:,boot_index[t]]
    
    # Compute covariances and matrix determinant based on resampled variables
    boot_C = np.cov(boot_x)
    boot_dC = np.empty_like(boot_C) * 0.
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            boot_dC[j,i] = (np.sum((boot_x[j,:] - np.nanmean(boot_x[j,:])) * (boot_dx[i,:] - np.nanmean(boot_dx[i,:])))) / (N - 1.)
    boot_detC = np.linalg.det(boot_C)
    
    # Compute cofactors based on resampled variables
    boot_Delta = np.linalg.inv(boot_C).T * boot_detC

    # Compute absolute transfer of information (T) and correlation coefficient based on resampled variables
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            boot_T[j,i] = compute_liang_index(boot_detC,boot_Delta[j,:],boot_dC[:,i],boot_C[i,j],boot_C[i,i])
    
    # Compute noise terms based on resampled variables
    boot_g = np.zeros(nvar)
    for i in np.arange(nvar):
        a1k = np.dot(np.linalg.inv(boot_C),boot_dC[:,i])
        f1 = np.nanmean(boot_dx[i,:])
        for k in np.arange(nvar):
            f1 = f1 - a1k[k] * np.nanmean(boot_x[k,:])
        R1 = boot_dx[i,:] - f1
        for k in np.arange(nvar):
            R1 = R1 - a1k[k] * boot_x[k,:]
        Q1 = np.sum(R1**2.)       
        boot_g[i] = Q1 * dt / N # equation (10)

    # Compute relative transfer of information (tau) based on resampled variables
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            boot_tau[j,i] = compute_liang_index_norm(boot_detC,boot_Delta[i,:],boot_dC[:,i],boot_T[:,i],boot_T[i,i],boot_g[i],boot_C[i,i],boot_T[j,i])
    
    # Return result of function
    return T,tau,boot_T,boot_tau