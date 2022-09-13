#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Fig. 3
    Map Liang index SSTt-THF
    Only 2 variables
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
    Computed via compute_liang_2var_2.py
    Dataset 1: J-OFURO3 (Japanese Ocean Flux Data Sets with Use of Remote-Sensing Observations, 0.25deg)
PROGRAMMER
    D. Docquier
LAST UPDATE
    25/08/2022
'''

# Standard libraries
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER,LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from statsmodels.stats.multitest import multipletests

# Options
save_var = True # True: computations + save variables; False: load variables
save_fig = True # True: save figures
alpha_fdr = 0.05 # alpha of FDR
nvar = 2 # number of variables
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
filename_liang = dir_output + 'SSTt_THF_Liang_2var_01.npy'
filename_sig = dir_output + 'SSTt_THF_Liang_2var_sig_fdr_alpha005.npy'

# Load Liang index and 1st boostraped value
boot_tau = np.zeros((ny,nx,n_iter,nvar,nvar),dtype='float32')
tau,boot_tau[:,:,0,:,:] = np.load(filename_liang,allow_pickle=True)
    
# Load and save variables
if save_var == True:
    
    # Load bootstraped values
    for i in np.arange(n_iter-1):
        print(i)
        if i < 8:
            filename = dir_output + 'SSTt_THF_Liang_2var_0' + str(i+2) + '.npy'
        else:
            filename = dir_output + 'SSTt_THF_Liang_2var_' + str(i+2) + '.npy'
        boot_tau[:,:,i+1,:,:] = np.load(filename,allow_pickle=True)
        
    # Compute p value of tau based on bootstrap distribution
    pval_tau = np.zeros((ny,nx,nvar,nvar))
    for y in np.arange(ny):
        print(y)
        for x in np.arange(nx):
            for i in np.arange(nvar):
                for j in np.arange(nvar):
                    pval_tau[y,x,i,j] = (np.count_nonzero((boot_tau[y,x,:,i,j]-tau[y,x,i,j]) >= np.abs(tau[y,x,i,j]))  \
                        + np.count_nonzero((boot_tau[y,x,:,i,j]-tau[y,x,i,j]) <= -np.abs(tau[y,x,i,j]))) / n_iter
    
    # Clear variables
    del boot_tau
    
    # Compute significance of Liang index (FDR)
    sig_tau_fdr = np.zeros((ny,nx,nvar,nvar))
    pval_tau_fdr = np.zeros((ny,nx,nvar,nvar))
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            pval_tau_1d = np.ravel(pval_tau[:,:,i,j])
            sig_tau_fdr_init = multipletests(pval_tau_1d,alpha=alpha_fdr,method='fdr_bh')[0]
            sig_tau_fdr_init[sig_tau_fdr_init==True] = 1
            sig_tau_fdr_init[sig_tau_fdr_init==False] = 0
            sig_tau_fdr[:,:,i,j] = np.reshape(sig_tau_fdr_init,(ny,nx))
    
    # Save variables
    np.save(filename_sig,[sig_tau_fdr])

else:
    
    # Load significance
    sig_tau_fdr = np.load(filename_sig,allow_pickle=True)[0]

# Cartopy projection
proj = ccrs.Mollweide()

# Palettes
palette_tau = plt.cm.seismic._resample(20)
min_tau = -30.
max_tau = 30.


########
# Maps #
########

# Fig. 3: Relative transfer of information SSTt-THF
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau SSTt->THF
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,tau[:,:,0,1],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax1.contour(lon,lat,sig_tau_fdr[:,:,0,1],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title(r'$\tau_{SSTt \longrightarrow THF}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{SSTt \longrightarrow THF}$ ($\%$)',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# tau THF->SSTt
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,tau[:,:,1,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax2.contour(lon,lat,sig_tau_fdr[:,:,1,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title(r'$\tau_{THF \longrightarrow SSTt}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{THF \longrightarrow SSTt}$ ($\%$)',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig3.png')