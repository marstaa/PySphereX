#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines calculus routines on a sphere.

Authors:
    Martin Staab <martin.staab@aei.mpg.de>
"""

import numpy as np


def sph_integrate(phi, theta, data):
    """Compute the surface integral over the unit sphere.

    Args:
        phi: array of azimuthal angles
        theta: array of polar angles
        data: 2-dim array of data
    """

    if phi.size != data.shape[1]:
        raise ValueError(f'phi and data[0,:] must be of same size (got {phi.size}, {data.shape[1]})')

    if theta.size != data.shape[0]:
        raise ValueError(f'theta and data[:,0] must be of same size (got {theta.size}, {data.shape[0]})')

    delta_phi = ((np.roll(phi, -1) - phi) + 2 * np.pi) % (2 * np.pi)
    phi = phi + delta_phi
    delta_phi = ((np.roll(phi, -1) - phi) + 2 * np.pi) % (2 * np.pi)

    delta_theta = np.diff(theta)
    theta = np.pad(theta[:-1] + delta_theta / 2, (1, 1), constant_values=(0, np.pi))
    delta_theta = np.diff(theta)

    return np.sum(data * delta_phi * ((np.sin(theta[:-1]) + np.sin(theta[1:])) / 2 * delta_theta)[:,None])
