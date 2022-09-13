#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Figs. 4-5 and S3 (J-OFURO3) + Figs. S4-S5 (SeaFlux)
    Map Liang index SST-THF and SSTt-THF
    Significance based on bootstrap resampling with replacement, based on boostrap distribution
    Computed via compute_liang.py
    Dataset 1: J-OFURO3 (Japanese Ocean Flux Data Sets with Use of Remote-Sensing Observations, 0.25deg)
    Dataset 2: SeaFlux (NASA Global Hydrology Resource Center, 0.25deg)
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
dataset = 1 # 1: J-OFURO3; 2: SeaFlux
alpha_fdr = 0.05 # alpha of FDR
nvar = 3 # number of variables
n_iter = 500 # number of bootstrap realizations

# Working directories
if dataset == 1:
    dir_input = '/ec/res4/hpcperm/cvaf/J-OFURO3/'
    dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/J-OFURO3/'
elif dataset == 2:
    dir_input = '/ec/res4/hpcperm/cvaf/SeaFlux/extracted/'
    dir_output = '/ec/res4/hpcperm/cvaf/ROADMAP/SeaFlux/'
dir_fig = '/perm/cvaf/ROADMAP/Air-Sea/figures/Observations/'

# Load latitude and longitude
if dataset == 1:
    filename = dir_input + 'SST/J-OFURO3_SST_V1.1_MONTHLY_HR_1988.nc'
    fh = Dataset(filename, mode='r')
    lat_init = fh.variables['latitude'][:]
    lon_init = fh.variables['longitude'][:]
elif dataset == 2:
    filename = dir_input + 'selection_SeaFluxV3_Monthly_1988.nc'
    fh = Dataset(filename, mode='r')
    lat_init = fh.variables['lat'][:]
    lon_init = fh.variables['lon'][:]
lon,lat = np.meshgrid(lon_init,lat_init)
fh.close()
ny,nx = lon.shape

# File names
if dataset == 1:
    filename_liang = dir_output + 'SST_THF_Liang_01.npy'
    filename_sig = dir_output + 'SST_THF_Liang_sig_fdr_alpha005.npy'
elif dataset == 2:
    filename_liang = dir_output + 'SST_THF_Liang_01_SeaFlux.npy'
    filename_sig = dir_output + 'SST_THF_Liang_sig_fdr_alpha005_SeaFlux.npy'
    
# Load and save variables
if save_var == True:
    
    # Load Liang index and 1st boostraped value
    boot_tau = np.zeros((ny,nx,n_iter,nvar,nvar),dtype='float16')
    tau,boot_tau[:,:,0,:,:] = np.load(filename_liang,allow_pickle=True)
    
    # Load bootstraped values
    for i in np.arange(n_iter-1):
        print(i)
        if i < 8:
            if dataset == 1:
                filename = dir_output + 'SST_THF_Liang_0' + str(i+2) + '.npy'
            elif dataset == 2:
                filename = dir_output + 'SST_THF_Liang_0' + str(i+2) + '_SeaFlux.npy'
        else:
            if dataset == 1:
                filename = dir_output + 'SST_THF_Liang_' + str(i+2) + '.npy'
            elif dataset == 2:
                filename = dir_output + 'SST_THF_Liang_' + str(i+2) + '_SeaFlux.npy'
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
    
    # Load Liang index
    tau,notused = np.load(filename_liang,allow_pickle=True)
    
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

# Fig. 4 / S4: Relative transfer of information SST-THF
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau SST->THF
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,tau[:,:,0,2],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax1.contour(lon,lat,sig_tau_fdr[:,:,0,2],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
if dataset == 1:
    ax1.set_title(r'$\tau_{SST \longrightarrow THF}$ 1988-2017 - J-OFURO3',fontsize=24)
elif dataset == 2:
    ax1.set_title(r'$\tau_{SST \longrightarrow THF}$ 1988-2017 - SeaFlux',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{SST \longrightarrow THF}$ ($\%$)',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# tau THF->SST
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,tau[:,:,2,0],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax2.contour(lon,lat,sig_tau_fdr[:,:,2,0],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
if dataset == 1:
    ax2.set_title(r'$\tau_{THF \longrightarrow SST}$ 1988-2017 - J-OFURO3',fontsize=24)
elif dataset == 2:
    ax2.set_title(r'$\tau_{THF \longrightarrow SST}$ 1988-2017 - SeaFlux',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{THF \longrightarrow SST}$ ($\%$)',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    if dataset == 1:
        fig.savefig(dir_fig + 'fig4.png')
    elif dataset == 2:
        fig.savefig(dir_fig + 'figS4.png')
        

# Fig. 5 / S5: Relative transfer of information SSTt-THF
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau SSTt->THF
ax1 = fig.add_subplot(2,1,1,projection=proj)
cs1 = ax1.pcolormesh(lon,lat,tau[:,:,1,2],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax1.contour(lon,lat,sig_tau_fdr[:,:,1,2],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax1.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
if dataset == 1:
    ax1.set_title(r'$\tau_{SSTt \longrightarrow THF}$ 1988-2017 - J-OFURO3',fontsize=24)
elif dataset == 2:
    ax1.set_title(r'$\tau_{SSTt \longrightarrow THF}$ 1988-2017 - SeaFlux',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{SSTt \longrightarrow THF}$ ($\%$)',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# tau THF->SSTt
ax2 = fig.add_subplot(2,1,2,projection=proj)
cs2 = ax2.pcolormesh(lon,lat,tau[:,:,2,1],cmap=palette_tau,vmin=min_tau,vmax=max_tau,transform=ccrs.PlateCarree())
ax2.contour(lon,lat,sig_tau_fdr[:,:,2,1],range(1,2,1),colors='black',linewidths=1,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND,zorder=1,edgecolor='k')
gl = ax2.gridlines(color='lightgray',linestyle='--',linewidth=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,86,30))
gl.yformatter = LATITUDE_FORMATTER
if dataset == 1:
    ax2.set_title(r'$\tau_{THF \longrightarrow SSTt}$ 1988-2017 - J-OFURO3',fontsize=24)
elif dataset == 2:
    ax2.set_title(r'$\tau_{THF \longrightarrow SSTt}$ 1988-2017 - SeaFlux',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{THF \longrightarrow SSTt}$ ($\%$)',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    if dataset == 1:
        fig.savefig(dir_fig + 'fig5.png')
    elif dataset == 2:
        fig.savefig(dir_fig + 'figS5.png')
        
        
# Fig. S3: Relative transfer of information SST-SSTt
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.85,top=0.93,wspace=None,hspace=0.15)

# tau SST->SSTt
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
if dataset == 1:
    ax1.set_title(r'$\tau_{SST \longrightarrow SSTt}$ 1988-2017 - J-OFURO3',fontsize=24)
elif dataset == 2:
    ax1.set_title(r'$\tau_{SST \longrightarrow SSTt}$ 1988-2017 - SeaFlux',fontsize=24)
cb_ax = fig.add_axes([0.87,0.6,0.02,0.3])
cbar = fig.colorbar(cs1,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{SST \longrightarrow SSTt}$ ($\%$)',fontsize=18)
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# tau SSTt->SST
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
if dataset == 1:
    ax2.set_title(r'$\tau_{SSTt \longrightarrow SST}$ 1988-2017 - J-OFURO3',fontsize=24)
elif dataset == 2:
    ax2.set_title(r'$\tau_{SSTt \longrightarrow SST}$ 1988-2017 - SeaFlux',fontsize=24)
cb_ax = fig.add_axes([0.87,0.1,0.02,0.3])
cbar = fig.colorbar(cs2,cax=cb_ax,orientation='vertical',ticks=[-30,-15,0,15,30],extend='both')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\tau_{SSTt \longrightarrow SST}$ ($\%$)',fontsize=18)
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    if dataset == 1:
        fig.savefig(dir_fig + 'figS3.png')
    elif dataset == 2:
        fig.savefig(dir_fig + 'SST_SSTt_Liang_rel_SeaFlux.png')