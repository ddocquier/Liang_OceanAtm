#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute correlation and covariance between SST / SSTt and turbulent heat flux (THF)
    Data: J-OFURO3 1988-2017
PROGRAMMER
    D. Docquier
LAST UPDATE
    25/08/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Options
save_var = True # True: save variables
nyears = int(2017-1988+1) # number of years
nmy = 12 # number of months in a year
nm = int(nyears * nmy) # number of months

# Working directories
dir_input = '/ec/res4/hpcperm/cvaf/J-OFURO3/'
dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/J-OFURO3/'
dir_fig = '/perm/cvaf/ROADMAP/Air-Sea/figures/'

# Load latitude and longitude
filename = dir_input + 'SST/J-OFURO3_SST_V1.1_MONTHLY_HR_1988.nc'
fh = Dataset(filename, mode='r')
lat_init = fh.variables['latitude'][:]
lon_init = fh.variables['longitude'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape

# File names
filename_cov = dir_output + 'SST_THF_covariance.npy'
filename_cor = dir_output + 'SST_THF_correlation.npy'

# Initialize variables (with zeroes)
sst = np.zeros((nm,ny,nx))
thf = np.zeros((nm,ny,nx))
sst_trend = np.zeros((nm,ny,nx))
covariance1 = np.zeros((ny,nx))
corrcoef1 = np.zeros((ny,nx))
pval1 = np.zeros((ny,nx))
covariance2 = np.zeros((ny,nx))
corrcoef2 = np.zeros((ny,nx))
pval2 = np.zeros((ny,nx))

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

# Remove trend and seasonality and compute correlation and covariance
for y in np.arange(ny):
    print(y)
    for x in np.arange(nx):
        if np.count_nonzero(np.isnan(sst[:,y,x])) >= 1 or np.count_nonzero(np.isnan(thf[:,y,x])) >= 1: # reject grid points with NaN values
            covariance1[y,x] = np.nan
            corrcoef1[y,x] = np.nan
            covariance2[y,x] = np.nan
            corrcoef2[y,x] = np.nan
        else:
            sst_trend[1:nm-1,y,x] = (sst[2:nm,y,x] - sst[0:nm-2,y,x]) / 2.
            sst_resid = seasonal_decompose(sst[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            sst_trend_resid = seasonal_decompose(sst_trend[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            thf_resid = seasonal_decompose(thf[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            covariance1[y,x] = np.cov(sst_resid,thf_resid)[0,1]
            corrcoef1[y,x],pval1[y,x] = stats.pearsonr(sst_resid,thf_resid)
            covariance2[y,x] = np.cov(sst_trend_resid,thf_resid)[0,1]
            corrcoef2[y,x],pval2[y,x] = stats.pearsonr(sst_trend_resid,thf_resid)
    
# Save variables
np.save(filename_cov,[covariance1,covariance2])
np.save(filename_cor,[corrcoef1,pval1,corrcoef2,pval2])