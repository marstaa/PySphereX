#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests `Expansion` class.

Authors:
    Martin Staab <martin.staab@aei.mpg.de>
"""

import numpy as np
from pytest import approx
from pyspherex import Expansion


def test_expansion_generate_sph_basis():
    """Check `Expansion.generate_sph_basis` against hard coded equivalent."""
    size_phi = 200
    size_theta = 100
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]

    result = Expansion.generate_sph_basis(phi, theta, 2)

    assert result[0][0] == np.sqrt(1 / 4 / np.pi)
    assert np.all(result[1][0] == np.sqrt(3 / 4 / np.pi) * np.cos(theta)[:,None])
    assert np.all(result[1][1] == -np.sqrt(3 / 2 / np.pi) / 2 * np.sin(theta)[:,None] * np.exp(1j * phi))
    assert np.all(result[2][0] == approx(np.sqrt(5 / np.pi) / 4 * (3 * np.cos(theta)[:,None]**2 - 1)))
    assert np.all(result[2][1] == approx(-np.sqrt(15 / 2 / np.pi) / 2 * (np.sin(theta) * np.cos(theta))[:,None] * np.exp(1j * phi)))
    assert np.all(result[2][2] == approx(np.sqrt(15 / 2 / np.pi) / 4 * np.sin(theta)[:,None]**2 * np.exp(2j * phi)))

def test_expansion_from_data_constant():
    """Spherical harmonics expansion of constant data"""
    size_phi = 2000
    size_theta = 1000
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]

    data = np.ones((theta.size, phi.size))
    expansion = Expansion.from_data(phi, theta, data, 1)

    assert expansion.coeffs[0][0] == approx(np.sqrt(4 * np.pi))
    assert expansion.coeffs[1][0] == approx(0)
    assert expansion.coeffs[1][1] == approx(0)
