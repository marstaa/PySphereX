#######################################################
# Spherical harmonics
# Example for the PySphereX package
# 2021 06 11
# author: prosku
#######################################################

#Load packages

import numpy as np
import matplotlib as matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib.pyplot as plt
import scipy.special

from netCDF4 import Dataset
import os
#import fnmatch
#import pandas as pd

#Plotting
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature

output_path = ''
variables = ['CDNC', 'ICNC']
varnames = ['CDNC', 'ICNC']
#variables = ['IWP', 'LWP', 'SCRE', 'LCRE', 'CC', 'Prcp_tot', 'CDNC', 'ICNC', 'LCC_MOD', 'ICC_MOD', 'MCC_MOD']
#varnames = ['IWP', 'LWP', 'SCRE', 'LCRE', 'CC', 'Prcp\_tot', 'CDNC', 'ICNC', 'LCC', 'ICC', 'MCC']

def sph_harm_coeff(data, phi, theta, l, m):
    """Calculate sperical harmonics coefficient
    
    Args:
        data (ndarray of dim 2): gridded data of shape (theta.size, phi.size)
        phi (ndarray): azimuthal angle
        theta (ndarray): polar angle
        l (int): degree of harmonic l>=0
        m (int): order of harmonic |m|<=l
        
    Return:
        coeff (float): spherical harmonics coefficient
    """
    dphi, dtheta = 2 * np.pi / data.shape[1], np.pi / data.shape[0]
    Y = scipy.special.sph_harm(m, l, phi[None,:], theta[:,None])
    return dphi * dtheta * np.sum(data * np.sin(theta[:,None]) * np.conj(Y))

def sph_harm_exp(data, phi, theta, L):
    """Calculate spherical harmonics coefficients for l
    
    Args:
        data (ndarray of dim 2): gridded data of shape (theta.size, phi.size)
        phi (ndarray): azimuthal angle
        theta (ndarray): polar angle
        L (int): maximum degree of harmonic L>=0
        
    Return:
        coeffs (ndarray): spherical harmonics coefficient |m|<=l
    """
    res = []
    for l in range(L + 1):
        coeffs = np.empty(2 * l + 1, dtype=np.complex128)
        for i in range(2 * l + 1):
            m = i - l
            coeffs[i] = sph_harm_coeff(data, phi, theta, l, m)
        res.append(coeffs)
    return res

def sph_harm_series(coeffs, phi, theta):
    """Reconstruct spherical harmonics series from coefficients
    
    Args:
        coeffs (list): list of coefficient arrays
        phi (ndarray): azimuthal angle
        theta (ndarray): polar angle
        
    Return:
        data (ndarray of dim 2): evaluated spherical harmonics expansion
    """
    L = len(coeffs)
    res = np.zeros((theta.size, phi.size), dtype=np.complex128)
    for l in range(L):
        for i in range(2 * l + 1):
            m = i - l
            Y = scipy.special.sph_harm(m, l, phi[None,:], theta[:,None])
            res += coeffs[l][i] * Y
    return np.real(res)

def sph_harm_spec(data, phi, theta, L):
    """Calculate power spectrum
    
    Args:
        data (ndarray of dim 2): gridded data of shape (theta.size, phi.size)
        phi (ndarray): azimuthal angle
        theta (ndarray): polar angle
        L (int): maximum degree of harmonic L>=0
        
    Return:
        l, spec (ndarray): degrees of harmonics, power spectrum
    """
    coeffs = sph_harm_exp(data, phi, theta, L)
    #res = np.array([np.sum(np.abs(c)**2) / (2 * l + 1) for l, c in enumerate(coeffs)])
    res = np.array([np.sum(np.abs(c)**2) for l, c in enumerate(coeffs)])
    return np.arange(L + 1), res

def sph_harm_norm(data, phi, theta):
    """Calculate sperical harmonics norm

    Args:
        data (ndarray of dim 2): gridded data of shape (theta.size, phi.size)
        phi (ndarray): azimuthal angle
        theta (ndarray): polar angle

    Return:
        coeff (float): spherical harmonics coefficient
    """
    dphi, dtheta = 2 * np.pi / data.shape[1], np.pi / data.shape[0]
    #Y = scipy.special.sph_harm(m, l, phi[None,:], theta[:,None])
    return np.sqrt(dphi * dtheta * np.sum(data * np.sin(theta[:,None]) * data))


def sph_harm_norm_ls(data_ls, phi, theta):
    """Calculate sperical harmonics norm from coefficients

    Args:
        data (ndarray of dim 2): gridded data of shape (theta.size, phi.size)
        phi (ndarray): azimuthal angle
        theta (ndarray): polar angle

    Return:
        coeff (float): spherical harmonics coefficient
    """
    coeffs_ls = sph_harm_exp(data_ls, phi, theta, 60)
    #dphi, dtheta = 2 * np.pi / data.shape[1], np.pi / data.shape[0]
    #Y = scipy.special.sph_harm(m, l, phi[None,:], theta[:,None])
    coeffs_squared = [f**2 for f in coeffs_ls]
    return np.sqrt(np.sum([np.sum(coeff) for coeff in coeffs_squared]))

def plot_sph_harm(var, ppe_IWP, ctrl_IWP):
    data=ppe_IWP[var][0,:,:] #- ctrl_IWP[var][0,:,:] #TODO
    coeffs = sph_harm_exp(data, phi, theta, 20)
    series = sph_harm_series(coeffs, phi, theta)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6*2,1*1.5), subplot_kw={'projection': ccrs.PlateCarree()})#, gridspec_kw={'hspace': 0.6})

    vmin=np.nanmin([data,series])
    vmax=np.nanmax([data,series])
    vext = np.nanmax([np.abs(vmin), np.abs(vmax)])
    print(np.nanmin(data), np.nanmax(data))
    if var=='CDNC':
        im = axs[0].pcolormesh(lons, lats, data, cmap='RdBu_r', norm=matplotlib.colors.LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)))
    else:
        im = axs[0].pcolormesh(lons, lats, data, cmap='RdBu_r', vmin=-vext, vmax=vext)
    axs[0].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='none'))
    #import IPython; IPython.embed()
    fig.colorbar(im, ax=axs[0], label=var)
    axs[0].set_xlabel('lon')
    axs[0].set_ylabel('lat')

    im = axs[1].pcolormesh(lons, lats, series, cmap='RdBu_r', vmin=-vext, vmax=vext)
    axs[1].set_xlabel('lon')
    axs[1].set_ylabel('lat')
    axs[1].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='none'))
    fig.colorbar(im, ax=axs[1])

    axs[0].set_title('Data')
    axs[1].set_title('Reconstructed')

    l, spec = sph_harm_spec(data, phi, theta, 20)
    spec2 = np.array([np.abs(c[l])**2 for l, c in enumerate(coeffs)])
    axs[2].remove()
    ax = fig.add_subplot(1,3,3)
    ax.plot(l, spec**(1/2), label='all m')
    ax.plot(l, spec2**(1/2), label='m=0 only')
    #plt.plot(l, 1e4/l**3)
    #plt.xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('l')
    ax.set_ylabel('$\sqrt{|f_l^m|^2}$')
    ax.legend()

    # Set spacing between plots
    plt.subplots_adjust(wspace=0.6)

    plt.savefig(output_path+'/geogr_'+var+'.pdf', bbox_inches='tight')

def plot_sph_harm_ls(data):
    coeffs = sph_harm_exp(data, phi, theta, 20)
    series = sph_harm_series(coeffs, phi, theta)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6*2,1*1.5), subplot_kw={'projection': ccrs.PlateCarree()})#, gridspec_kw={'hspace': 0.8})

    vmin=np.nanmin([data,series])
    vmax=np.nanmax([data,series])
    im = axs[0].pcolormesh(lons, lats, data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[0].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='none'))
    fig.colorbar(im, ax=axs[0])

    im = axs[1].pcolormesh(lons, lats, series, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[1].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='none'))
    fig.colorbar(im, ax=axs[1])

    axs[0].set_title('Data')
    axs[1].set_title('Reconstructed')

    axs[2].remove()
    ax = fig.add_subplot(1,3,3)
    l, spec = sph_harm_spec(data, phi, theta, 20)
    spec2 = np.array([np.abs(c[l])**2 for l, c in enumerate(coeffs)])
    ax.plot(l, spec**(1/2), label='all m')
    ax.plot(l, spec2**(1/2), label='m=0 only')
    #plt.plot(l, 1e4/l**3)
    #plt.xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('l')
    ax.set_ylabel('$\sqrt{|g_l^m|^2}$')
    ax.legend()

    # Set spacing between plots
    plt.subplots_adjust(wspace=0.6)

    plt.savefig(output_path+'/geogr_ls.pdf', bbox_inches='tight')

def alpha_sph_harm_ls(data, data_ls):
    # This is the correct way to get a normed land-sea mask
    dphi, dtheta = 2 * np.pi / data.shape[1], np.pi / data.shape[0]
    data_ls_normed = (data_ls - dphi*dtheta/(4*np.pi)*np.sum(data_ls*np.sin(theta[:,None])))
    data_ls_normed = data_ls_normed/sph_harm_norm_ls(data_ls_normed, phi, theta)
    coeffs_ls_normed = sph_harm_exp(data_ls_normed, phi, theta, 60)
    series_ls_normed = sph_harm_series(coeffs_ls_normed, phi, theta)
    coeffs = sph_harm_exp(data, phi, theta, 20)
    coeffs_f_g = [f*g for f, g in zip(coeffs, coeffs_ls_normed)]
    alpha = np.sum([np.sum(coeff) for coeff in coeffs_f_g])
    norm = sph_harm_norm(data, phi, theta)
    norm_ls = sph_harm_norm_ls(data_ls_normed, phi, theta)
    print(norm, norm_ls, alpha, alpha/(norm*norm_ls))


if __name__ == '__main__':
    lons = np.load('data/lons.npy')
    lats = np.load('data/lats.npy')

    phi, theta = lons / 180 * np.pi, - lats / 180 * np.pi + np.pi / 2
    dphi, dtheta = 1.875 * 180 / np.pi, 1.875 * 180 / np.pi

    data_ppe = Dataset('')
    data_ctrl = Dataset('')
    data_ls = Dataset('data/unit.24')['SLF'][:,:]

    for var, varnames in zip(variables, varnames):
       plot_sph_harm(var, data_ppe, data_ctrl) 
       print(var)
       #alpha_sph_harm_ls(data_ppe[var][0,:,:] - data_ctrl[var][0,:,:], data_ls) #TODO
       alpha_sph_harm_ls(data_ppe[var][0,:,:], data_ls)

    plot_sph_harm_ls(data_ls)

