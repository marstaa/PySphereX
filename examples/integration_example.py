#! /usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# -*- coding: utf-8 -*-
"""
Usage example for the spherical harmonics expansion in the PySphereX package

Authors:
    Ulrike Proske <ulrike.proske@env.ethz.ch>, 2021-08
"""

#Load packages

import numpy as np

# Read in netCDF data
from netCDF4 import Dataset

# Plotting
import matplotlib
import matplotlib.pyplot as plt

#PySphereX
from pyspherex import Expansion

DATA_PATH = 'example_data/IWP_non-meaningful-data.nc'
VAR_NAME = 'IWP' # variable name
VAR_UNIT = 'g/m²' # unit of that variable
DEGREE_MAX = 20

# pylint: disable=too-many-arguments, too-many-locals
def plot_sph_harm(var, unit, data, degree_max, lons, lats):
    """Plot data, spherical harmonics expansion of data and power spectrum.

    Args:
        var: variable name of the field to be plotted, must be present in the input data
        unit: unit of that variable as string
        data: netcdf data that contains the input field
    """
    data=data[var][0,:,:]

    # convert longitudes and latitudes to angles
    phi, theta = lons / 180 * np.pi, -lats / 180 * np.pi + np.pi / 2

    result = Expansion.from_data(phi, theta, data, degree_max)

    data_sh = np.real(result(phi, theta))

    fig = plt.figure(figsize=(3.27,3.27*2))
    plt.subplots_adjust(0,0,1,1)

    vext = np.nanmax([np.abs(np.nanmin([data,data_sh])), np.abs(np.nanmax([data,data_sh]))])/2 # make colors stronger
    ax1 = fig.add_subplot(311)

    if not np.any(data < 0):
        image = ax1.pcolormesh(lons, lats, data, cmap='RdBu_r',
                norm=matplotlib.colors.LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)),
                shading='auto')
    else:
        image = ax1.pcolormesh(lons, lats, data, cmap='RdBu_r', vmin=-vext, vmax=vext)
    # axis labels
    ax1.set_xlabel(r'Longitude')
    ax1.set_ylabel(r'Latitude')
    ax1.set_yticks([-60,-30,0,30,60])

    ax2 = fig.add_subplot(312)
    if not np.any(data < 0):
        image = ax2.pcolormesh(lons, lats, data_sh, cmap='RdBu_r',
                norm=matplotlib.colors.LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)),
                shading='auto')
    else:
        image = ax2.pcolormesh(lons, lats, data_sh, cmap='RdBu_r', vmin=-vext, vmax=vext)

    # axis labels
    ax2.set_xlabel(r'Longitude')
    ax2.set_ylabel(r'Latitude')
    ax2.set_yticks([-60,-30,0,30,60])
    fig.colorbar(image, ax=[ax1, ax2], label=var+' ('+unit+')', shrink=0.8, anchor=(2.5,0.5))

    ax1.set_title('Data')
    ax2.set_title('Reconstructed')

    spec2 = np.zeros((len(result.coeffs)))
    for degree in result.coeffs:
        spec2[degree] = np.abs(result.coeffs[degree][degree])**2 / 4 / np.pi
    ax = fig.add_subplot(313)
    degs, spec = result.spectrum
    ax.plot(degs, np.sqrt(spec), label='all orders', color='black')
    ax.plot(result.spectrum[0], spec2**(1/2), label='order zero only', color='grey')
    ax.set_yscale('log')
    ax.set_xlabel('degree')
    ax.set_ylabel(r'Angular amplitude spectrum')
    ax.legend()

    # Set spacing between plots
    plt.subplots_adjust(hspace=0.5)

    plt.savefig('geogr_'+var+'.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    longitudes = np.load('example_data/lons.npy')
    latitudes = np.load('example_data/lats.npy')

    data_input = Dataset(DATA_PATH)

    plot_sph_harm(VAR_NAME, VAR_UNIT, data_input, DEGREE_MAX, longitudes, latitudes)
