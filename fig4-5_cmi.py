#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Figs. 4-5
    Map CMI SST-THF given SSTt and SSTt-THF given SST
    Computed via compute_cmi_3var.py
    Dataset 1: J-OFURO3 (Japanese Ocean Flux Data Sets with Use of Remote-Sensing Observations, 0.25deg)
PROGRAMMER
    D. Docquier
LAST UPDATE
    19/12/2022
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
save_var = False # True: computations + save variables; False: load variables
save_fig = True # True: save figures
alpha_fdr = 0.05 # alpha of FDR
nvar = 3 # number of variables
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
filename_cmi = dir_output + 'SST_THF_SSTt_CMI_01.npy'
#filename_cmi = dir_output + 'SST_THF_SSTt_CMI2.npy'

# Load CMI
cmi = np.load(filename_cmi,allow_pickle=True)[0]

# Cartopy projection
proj = ccrs.Mollweide()

# Palettes
palette_cmi = plt.cm.Reds._resample(20)
#min_cmi = np.nanmin(cmi[:,:,0,1])
#max_cmi = np.nanmax(cmi[:,:,0,1])
#min_cmi2 = np.nanmin(cmi[:,:,1,2])
#max_cmi2 = np.nanmax(cmi[:,:,1,2])
min_cmi = 0
max_cmi = 0.7
min_cmi2 = 0
max_cmi2 = 0.25


########
# Maps #
########

# Fig. 4: Relative transfer of information SST-THF
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# CMI SST->THF
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,cmi[:,:,0,1],cmap=palette_cmi,vmin=min_cmi,vmax=max_cmi,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title(r'$CMI_{SST \longrightarrow THF | SSTt}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[min_cmi,(max_cmi-min_cmi)/4.,(max_cmi-min_cmi)/2.,3.*(max_cmi-min_cmi)/4.,max_cmi],extend='max')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$CMI_{SST \longrightarrow THF}$',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# CMI THF->SST
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,cmi[:,:,1,0],cmap=palette_cmi,vmin=min_cmi,vmax=max_cmi,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title(r'$CMI_{THF \longrightarrow SST | SSTt}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[min_cmi,(max_cmi-min_cmi)/4.,(max_cmi-min_cmi)/2.,3.*(max_cmi-min_cmi)/4.,max_cmi],extend='max')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$CMI_{THF \longrightarrow SST}$',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig4_cmi.png')
    
    
# Fig. 5: Relative transfer of information SSTt-THF
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# CMI SSTt->THF
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,cmi[:,:,2,1],cmap=palette_cmi,vmin=min_cmi2,vmax=max_cmi2,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax1.set_title(r'$CMI_{SSTt \longrightarrow THF | SST}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[min_cmi2,(max_cmi2-min_cmi2)/4.,(max_cmi2-min_cmi2)/2.,3.*(max_cmi2-min_cmi2)/4.,max_cmi2],extend='max')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$CMI_{SSTt \longrightarrow THF}$',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# CMI THF->SSTt
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,cmi[:,:,1,2],cmap=palette_cmi,vmin=min_cmi2,vmax=max_cmi2,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
ax2.set_title(r'$CMI_{THF \longrightarrow SSTt | SST}$ 1988-2017 - J-OFURO3',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[min_cmi2,(max_cmi2-min_cmi2)/4.,(max_cmi2-min_cmi2)/2.,3.*(max_cmi2-min_cmi2)/4.,max_cmi2],extend='max')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$CMI_{THF \longrightarrow SSTt}$',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig5_cmi.png')