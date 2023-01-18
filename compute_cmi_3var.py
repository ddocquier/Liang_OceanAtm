#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Compute CMI SST-THF given SSTt and SSTt-THF given THF
    J-OFURO3 (Japanese Ocean Flux Data Sets with Use of Remote-Sensing Observations, 0.25deg)
PROGRAMMER
    D. Docquier
LAST UPDATE
    15/12/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import knncmi # library created by O. Mesner to compute CMI (https://github.com/omesner/knncmi)
import pandas as pd

# Options
boot_iter = int(sys.argv[1])
nyears = int(2017-1988+1) # number of years
nmy = 12 # number of months in a year
nm = int(nyears * nmy) # number of months
nvar = 3 # number of variables
dt = 1 # time step
hyper_nn = 3 # hyperparameter for the kth nearest neighbor (default value = 3)

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

# File names
if boot_iter < 10:
    filename_cmi = dir_output + 'SST_THF_SSTt_CMI_0' + str(boot_iter) + '.npy'
else:
    filename_cmi = dir_output + 'SST_THF_SSTt_CMI_' + str(boot_iter) + '.npy'

# Initialize variables (with zeroes)
sst = np.zeros((nm,ny,nx),dtype='float32')
thf = np.zeros((nm,ny,nx),dtype='float32')
sst_trend = np.zeros((nm,ny,nx),dtype='float32')
cmi_val = np.zeros((ny,nx,nvar,nvar))
boot_cmi = np.zeros((ny,nx,nvar,nvar))

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

# Remove trend and seasonality of SST and THF and compute CMI in each grid point
for y in np.arange(ny):
    print(y)
    for x in np.arange(nx):
        if np.count_nonzero(np.isnan(sst[:,y,x])) >= 1 or np.count_nonzero(np.isnan(thf[:,y,x])) >= 1:
            cmi_val[y,x,:,:] = np.nan
        else:
            sst_trend[1:nm-1,y,x] = (sst[2:nm,y,x] - sst[0:nm-2,y,x]) / 2. # central difference approximation
            sst_resid = seasonal_decompose(sst[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            sst_trend_resid = seasonal_decompose(sst_trend[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            thf_resid = seasonal_decompose(thf[:,y,x],model='additive',period=12,extrapolate_trend='freq').resid
            xx = np.array((sst_resid,thf_resid,sst_trend_resid))
            xx_pd = pd.DataFrame(xx.T,columns=['SST','THF','SSTt']) # convert numpy array into panda data frame
            cmi_val[y,x,0,1] = knncmi.cmi(['SST'],['THF'],['SSTt'],hyper_nn,xx_pd) # I(SST;THF|SSTt)
            xx2 = np.array((thf_resid,sst_resid,sst_trend_resid))
            xx_pd2 = pd.DataFrame(xx2.T,columns=['THF','SST','SSTt'])
            cmi_val[y,x,1,0] = knncmi.cmi(['THF'],['SST'],['SSTt'],hyper_nn,xx_pd2) # I(THF;SST|SSTt)
            xx3 = np.array((sst_trend_resid,thf_resid,sst_resid))
            xx_pd3 = pd.DataFrame(xx3.T,columns=['SSTt','THF','SST'])
            cmi_val[y,x,2,1] = knncmi.cmi(['SSTt'],['THF'],['SST'],hyper_nn,xx_pd3) # I(SSTt;THF|SST)
            xx4 = np.array((thf_resid,sst_trend_resid,sst_resid))
            xx_pd4 = pd.DataFrame(xx4.T,columns=['THF','SSTt','SST'])
            cmi_val[y,x,1,2] = knncmi.cmi(['THF'],['SSTt'],['SST'],hyper_nn,xx_pd4) # I(THF;SSTt|SST)
    
# Save variables
if boot_iter == 1:
#    np.save(filename_cmi,[cmi_val,boot_cmi])
    np.save(filename_cmi,[cmi_val])
else:
    np.save(filename_cmi,[boot_cmi])