#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Figs. 6 and S6
    Map Liang index SST-THF(-1) and SSTt-THF(-1)
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
    Computed via compute_liang_lag.py and compute_fdr_liang_lag.py (2 slices for memory reasons)
    Data: J-OFURO3 (Japanese Ocean Flux Data Sets with Use of Remote-Sensing Observations, 0.25deg)
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

# Options
save_fig = True # True: save figures
shift = 1 # 1: shift 1 month before; -1: shift 1 month after
nvar = 4 # number of variables

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

# Load Liang index
filename_liang = dir_output + 'SST_THF_Liang_lag' + str(shift) + '_4var_01.npy'
tau,notused = np.load(filename_liang,allow_pickle=True)

# Load significance slice 1
filename_sig1 = dir_output + 'SST_THF_Liang_lag' + str(shift) + '_4var_sig_fdr_alpha005_1.npy'
sig_tau_fdr1 = np.load(filename_sig1,allow_pickle=True)[0]

# Load significance slice 2
filename_sig2 = dir_output + 'SST_THF_Liang_lag' + str(shift) + '_4var_sig_fdr_alpha005_2.npy'
sig_tau_fdr2 = np.load(filename_sig2,allow_pickle=True)[0]

# Concatenate datasets
sig_tau_fdr = np.concatenate((sig_tau_fdr1,sig_tau_fdr2),axis=0)

# Cartopy projection
proj = ccrs.Mollweide()

# Palettes
palette_tau = plt.cm.seismic._resample(20)
min_tau = -30.
max_tau = 30.


########
# Maps #
########  
        
# Relative transfer of information THF(-1)->SST and THF(-1)->SSTt  
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau THF(-1)->SST
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,tau[:,:,3,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax1.contour(lon,lat,sig_tau_fdr[:,:,3,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title(r'$\tau_{THF(-1) \longrightarrow SST}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{THF(-1) \longrightarrow SST}$ ($\%$)',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# tau THF(-1)->SSTt
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,tau[:,:,3,1],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax2.contour(lon,lat,sig_tau_fdr[:,:,3,1],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title(r'$\tau_{THF(-1) \longrightarrow SSTt}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{THF(-1) \longrightarrow SSTt}$ ($\%$)',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig6.png')
    
    
# Relative transfer of information SST->THF(-1) and SSTt->THF(-1) 
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau SST->THF(-1)
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,tau[:,:,0,3],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax1.contour(lon,lat,sig_tau_fdr[:,:,0,3],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title(r'$\tau_{SST \longrightarrow THF(-1)}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{SST \longrightarrow THF(-1)}$ ($\%$)',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# tau SSTt->THF(-1)
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,tau[:,:,1,3],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax2.contour(lon,lat,sig_tau_fdr[:,:,1,3],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title(r'$\tau_{SSTt \longrightarrow THF(-1)}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{SSTt \longrightarrow THF(-1)}$ ($\%$)',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'figS6.png')