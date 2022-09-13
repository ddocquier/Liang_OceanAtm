#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Figs. 1 (correlation) and S2 (covariance)
    Map correlation and covariance between SST / SSTt and turbulent heat flux (THF)
    Computed via compute_correlation.py
    Data: J-OFURO3 1988-2017
PROGRAMMER
    D. Docquier
LAST UPDATE
    02/09/2022
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
save_fig = True # True: save figures
alpha_fdr = 0.05 # alpha of FDR
nyears = int(2017-1988+1) # number of years
nmy = 12 # number of months in a year
nm = int(nyears * nmy) # number of months

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

# Load SST / SSTt - THF correlations
filename = dir_output + 'SST_THF_correlation.npy'
R_SST_THF,pval_SST_THF,R_SSTt_THF,pval_SSTt_THF = np.load(filename,allow_pickle=True)
ny,nx = R_SST_THF.shape

# Load SST / SSTt - THF covariances
filename = dir_output + 'SST_THF_covariance.npy'
cov_SST_THF,cov_SSTt_THF = np.load(filename,allow_pickle=True)

# Compute significance of SST-THF correlation (False Discovery Rate)
pval_SST_THF_1d = np.ravel(pval_SST_THF)
sig_SST_THF_1d = multipletests(pval_SST_THF_1d,alpha=alpha_fdr,method='fdr_bh')[0]
sig_SST_THF_1d[sig_SST_THF_1d==True] = 1
sig_SST_THF_1d[sig_SST_THF_1d==False] = 0
sig_SST_THF = np.reshape(sig_SST_THF_1d,(ny,nx))

# Compute significance of SSTt-THF correlation (False Discovery Rate)
pval_SSTt_THF_1d = np.ravel(pval_SSTt_THF)
sig_SSTt_THF_1d = multipletests(pval_SSTt_THF_1d,alpha=alpha_fdr,method='fdr_bh')[0]
sig_SSTt_THF_1d[sig_SSTt_THF_1d==True] = 1
sig_SSTt_THF_1d[sig_SSTt_THF_1d==False] = 0
sig_SSTt_THF = np.reshape(sig_SSTt_THF_1d,(ny,nx))

# Cartopy projection
proj = ccrs.Mollweide()

# Palettes
palette_cor = plt.cm.seismic._resample(20)
min_cor = -1.
max_cor = 1.
min_cov1 = -30.
max_cov1 = 30.
min_cov2 = -10.
max_cov2 = 10.


########
# Maps #
########
  
# Fig. 1: Correlation coefficient
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# SST-THF
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,R_SST_THF,cmap=palette_cor,vmin=min_cor,vmax=max_cor,transform=ccrs.PlateCarree())
ax1.contour(lon,lat,sig_SST_THF,range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title('SST-THF correlation 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-1,-0.5,0,0.5,1])
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Correlation coefficient',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# SSTt-THF
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,R_SSTt_THF,cmap=palette_cor,vmin=min_cor,vmax=max_cor,transform=ccrs.PlateCarree())
ax2.contour(lon,lat,sig_SSTt_THF,range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title('SSTt-THF correlation 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-1,-0.5,0,0.5,1])
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Correlation coefficient',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig1.png')
    
    
# Fig. S2: Covariance
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# SST-THF covariance
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,cov_SST_THF,cmap=palette_cor,vmin=min_cov1,vmax=max_cov1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title('SST-THF covariance 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Covariance ($^\circ$C W m$^{-2}$)',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# SSTt-THF covariance
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,cov_SSTt_THF,cmap=palette_cor,vmin=min_cov2,vmax=max_cov2,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title('SSTt-THF covariance 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-10,-5,0,5,10],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Covariance ($^\circ$C W m$^{-2}$ month$^{-1}$)',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'figS2.png')