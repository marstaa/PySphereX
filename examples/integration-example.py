#######################################################
# Spherical harmonics
# Example for the PySphereX package
# 2021 06 11
# author: prosku
# License? #TODO
#######################################################

#Load packages

import numpy as np
import matplotlib as matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import os
#import fnmatch
#import pandas as pd

#Plotting
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

#PySphereX
from pyspherex import Expansion
import pyspherex.calculus

plt.style.use('pyspherex_example.mplstyle')
var = 'IWP'
unit = '\SI{}{\g \per \square \meter}'
degree_max = 20

def plot_sph_harm(var, unit, data):
    data=data[var][0,:,:]
    result = Expansion.from_data(phi, theta, data, degree_max)

    # make it see the data as continuous
    data, lon = add_cyclic_point(data, coord=lons)
    data_sh, lon = add_cyclic_point(np.real(result(phi, theta)), coord=lons)

    fig = plt.figure(figsize=(3.27,3.27*2))
    plt.subplots_adjust(0,0,1,1)

    vmin=np.nanmin([data,data_sh])
    vmax=np.nanmax([data,data_sh])
    vext = np.nanmax([np.abs(vmin), np.abs(vmax)])/2 # make colors stronger
    ax1 = fig.add_subplot(311, projection=ccrs.PlateCarree())
    # https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis

    print('fine')

    if not np.any(data < 0):
        im = ax1.pcolormesh(lon, lats, data, cmap='RdBu_r', norm=matplotlib.colors.LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)))
    else:
        im = ax1.pcolormesh(lon, lats, data, cmap='RdBu_r', vmin=-vext, vmax=vext)
    print('fine')
    ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='none'))
    print('fine')
    ax1.set_xlabel('lon')
    ax1.set_ylabel('lat')
    # axis labels
    print('fine')
    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.5)
    print('fine')
    gl1.xlabels_top = False
    gl1.ylabels_right = False
    gl1.xlabels_bottom = False
    gl1.ylabels_left = False
    ax1.set_xlabel(r'Longitude ($^\circ$E)')
    ax1.set_ylabel(r'Latitude ($^\circ$N)')
    ax1.set_yticks([-60,-30,0,30,60])
    ax1.set_xticks([-120,-60,0,60,120])

    ax2 = fig.add_subplot(312, projection=ccrs.PlateCarree())#axs[1]
    if not np.any(data < 0):
        im = ax2.pcolormesh(lon, lats, data_sh, cmap='RdBu_r', norm=matplotlib.colors.LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)))
    else:
        im = ax2.pcolormesh(lon, lats, data_sh, cmap='RdBu_r', vmin=-vext, vmax=vext)
    ax2.set_xlabel('lon')
    ax2.set_ylabel('lat')
    ax2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='none'))

    # axis labels
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.5)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabels_bottom = False
    gl.ylabels_left = False

    ax2.set_xlabel(r'Longitude ($^\circ$E)')
    ax2.set_ylabel(r'Latitude ($^\circ$N)')
    ax2.set_yticks([-60,-30,0,30,60])
    ax2.set_xticks([-120,-60,0,60,120])
    fig.colorbar(im, ax=[ax1, ax2], label=var+' ('+unit+')', shrink=0.8, anchor=(2.5,0.5))

    ax1.set_title('Data')
    ax2.set_title('Reconstructed')

    spec2 = np.zeros((len(result.coeffs)))
    for l in result.coeffs:
        spec2[l] = np.abs(result.coeffs[l][l])**2 / 4 / np.pi
    #axs[2].remove()
    ax = fig.add_subplot(313)
    #ax = fig.add_subplot(2,2,(3,4))
    # https://scientificallysound.org/2017/09/07/matplotlib-subplots/
    ax.plot(result.spectrum[0], result.spectrum[1]**(1/2), label='all $m$', color='black')
    ax.plot(result.spectrum[0], spec2**(1/2), label='$m=0$ only', color='grey')
    ax.set_yscale('log')
    ax.set_xlabel('$l$')
    ax.set_ylabel('$\sqrt{S_{ff}}$')
    ax.legend()

    # Set spacing between plots
    plt.subplots_adjust(wspace=0.6)
    
    plt.savefig('geogr_'+var+'.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    lons = np.load('example_data/lons.npy')
    lats = np.load('example_data/lats.npy')

    phi, theta = lons / 180 * np.pi, - lats / 180 * np.pi + np.pi / 2
    dphi, dtheta = 1.875 * 180 / np.pi, 1.875 * 180 / np.pi

    data = Dataset('example_data/IWP_non-meaningful-data.nc')

    plot_sph_harm(var, unit, data)

