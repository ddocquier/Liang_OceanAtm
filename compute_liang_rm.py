#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute Liang index SST-THF and SSTt-THF
    With running mean
    Error based on bootstrap resampling
    J-OFURO3 (Japanese Ocean Flux Data Sets with Use of Remote-Sensing Observations, 0.25deg)
PROGRAMMER
    D. Docquier
LAST UPDATE
    25/08/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import seasonal_decompose
import sys

# Import my functions
sys.path.append('/home//cvaf/Air-Sea/')
from function_liang_nvar2 import compute_liang_nvar

# Function to compute running mean
def running_mean(var,N):
    rm = np.convolve(var,np.ones(N)/N,mode='same')
    return rm

# Options
boot_iter = int(sys.argv[1])
nyears = int(2017-1988+1) # number of years
nmy = 12 # number of months in a year
nm = int(nyears * nmy) # number of months
nvar = 3 # number of variables
dt = 1 # time step
time_rm = 12 # time for running mean (12 or 120 months)

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/J-OFURO3/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/J-OFURO3/'

# Load latitude and longitude
filename = dir_input + 'SST/J-OFURO3_SST_V1.1_MONTHLY_HR_1988.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['latitude'][:]
lon_init = fh.variables['longitude'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape

# File name
if boot_iter < 10:
    filename_liang = dir_output + 'SST_THF_Liang_rm' + str(time_rm) + '_0' + str(boot_iter) + '.npy'
else:
    filename_liang = dir_output + 'SST_THF_Liang_rm' + str(time_rm) + '_' + str(boot_iter) + '.npy'

# Initialize variables (with zeroes)
sst = np.zeros((nm,ny,nx))
thf = np.zeros((nm,ny,nx))
sst_trend = np.zeros((nm,ny,nx))
tau = np.zeros((ny,nx,nvar,nvar))
boot_tau = np.zeros((ny,nx,nvar,nvar))

# Loop over years
for year in np.arange(nyears):
    print(1988+year)

    # Load SST
    filename = dir_input + 'SST/J-OFURO3_SST_V1.1_MONTHLY_HR_' + str(1988+year) + '.nc'
    fh = Dataset(filename, mode='r')
    sst[year*nmy:year*nmy+nmy,:,:] = fh.variables['SST'][:]
    sst[sst<-100.] = np.nan
    fh.close()

    # Load LHF
    filename = dir_input + 'LHF/J-OFURO3_LHF_V1.1_MONTHLY_HR_' + str(1988+year) + '.nc'
    fh = Dataset(filename, mode='r')
    lhf = fh.variables['LHF'][:]
    lhf[lhf<-2000.] = np.nan
    fh.close()

    # Load SHF
    filename = dir_input + 'SHF/J-OFURO3_SHF_V1.1_MONTHLY_HR_' + str(1988+year) + '.nc'
    fh = Dataset(filename, mode='r')
    shf = fh.variables['SHF'][:]
    shf[shf<-2000.] = np.nan
    fh.close()
    
    # Compute turbulent heat flux (THF)
    thf[year*nmy:year*nmy+nmy,:,:] = lhf + shf
    
    del lhf,shf

# Remove trend and seasonality of SST and THF, compute SST trend and compute Liang index in each grid point
for y in np.arange(ny):
    print(y)
    for x in np.arange(nx):
        if np.count_nonzero(np.isnan(sst[:,y,x])) >= 1 or np.count_nonzero(np.isnan(thf[:,y,x])) >= 1:
            tau[y,x,:,:] = np.nan
        else:
            sst_trend[1:nm-1,y,x] = (sst[2:nm,y,x] - sst[0:nm-2,y,x]) / (2.*dt) # central difference approximation
            sst_resid = seasonal_decompose(sst[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            sst_trend_resid = seasonal_decompose(sst_trend[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            thf_resid = seasonal_decompose(thf[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            sst_resid = running_mean(sst_resid,time_rm)
            sst_trend_resid = running_mean(sst_trend_resid,time_rm)
            thf_resid = running_mean(thf_resid,time_rm)
            xx = np.array((sst_resid,sst_trend_resid,thf_resid))
            notused,tau[y,x,:,:],notused,boot_tau[y,x,:,:] = compute_liang_nvar(xx,dt)
    
# Save variables
if boot_iter == 1:
    np.save(filename_liang,[tau,boot_tau])
else:
    np.save(filename_liang,[boot_tau])