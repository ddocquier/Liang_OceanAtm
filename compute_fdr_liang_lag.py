#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute FDR significance of Liang index
    With 4 variables: SST, SSTt, THF, THF(-1)
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
PROGRAMMER
    D. Docquier
LAST UPDATE
    25/08/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.stats.multitest import multipletests

# Options
slice_y = 1
first_slice = 360
alpha_fdr = 0.05 # alpha of FDR
shift = 1 # 1: shift 1 month before; -1: shift 1 month after
nvar = 4 # number of variables
n_iter = 500 # number of bootstrap realizations

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/J-OFURO3/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/J-OFURO3/'
dir_fig = '/perm/cvaf/ROADMAP/Air-Sea/figures/Observations/'

# Load latitude and longitude
filename = dir_input + 'SST/J-OFURO3_SST_V1.1_MONTHLY_HR_1988.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['latitude'][:]
lon_init = fh.variables['longitude'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape
    
# File names
filename_liang = dir_output + 'SST_THF_Liang_lag' + str(shift) + '_4var_01.npy'
filename_sig = dir_output + 'SST_THF_Liang_lag' + str(shift) + '_4var_sig_fdr_alpha005_' + str(slice_y) + '.npy'

# Load Liang index and 1st bootstrapped value
if slice_y == 1:
    ny_slice = first_slice
    start_slice = int(0)
    end_slice = int(ny_slice)
elif slice_y == 2:
    ny_slice = int(ny - first_slice)
    start_slice = int(first_slice)
    end_slice = int(ny)
boot_tau = np.zeros((ny_slice,nx,n_iter,nvar,nvar),dtype='float16')
tau_init,boot_init = np.load(filename_liang,allow_pickle=True)
tau = tau_init[start_slice:end_slice,:,:,:]
boot_tau[:,:,0,:,:] = boot_init[start_slice:end_slice,:,:,:]
del tau_init,boot_init

# Load bootstraped values
for i in np.arange(n_iter-1):
    print(i)
    if i < 8:
        filename = dir_output + 'SST_THF_Liang_lag' + str(shift) + '_4var_0' + str(i+2) + '.npy'
    else:
        filename = dir_output + 'SST_THF_Liang_lag' + str(shift) + '_4var_' + str(i+2) + '.npy'
    boot_init = np.load(filename,allow_pickle=True)[0]
    boot_tau[:,:,i+1,:,:] = boot_init[start_slice:end_slice,:,:,:]
    del boot_init
    
# Compute p value of T and tau
pval_tau = np.zeros((ny_slice,nx,nvar,nvar))
for y in np.arange(ny_slice):
    print(y)
    for x in np.arange(nx):
        for i in np.arange(nvar):
            for j in np.arange(nvar):
                pval_tau[y,x,i,j] = (np.count_nonzero((boot_tau[y,x,:,i,j]-tau[y,x,i,j]) >= np.abs(tau[y,x,i,j]))  \
                    + np.count_nonzero((boot_tau[y,x,:,i,j]-tau[y,x,i,j]) <= -np.abs(tau[y,x,i,j]))) / n_iter

# Clear variables
del boot_tau

# Compute FDR
sig_tau_fdr = np.zeros((ny_slice,nx,nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        pval_tau_1d = np.ravel(pval_tau[:,:,i,j])
        sig_tau_fdr_init = multipletests(pval_tau_1d,alpha=alpha_fdr,method='fdr_bh')[0]
        sig_tau_fdr_init[sig_tau_fdr_init==True] = 1
        sig_tau_fdr_init[sig_tau_fdr_init==False] = 0
        sig_tau_fdr[:,:,i,j] = np.reshape(sig_tau_fdr_init,(ny_slice,nx))

# Save variables
np.save(filename_sig,[sig_tau_fdr])