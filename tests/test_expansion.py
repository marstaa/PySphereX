#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests `Expansion` class.

Authors:
    Martin Staab <martin.staab@aei.mpg.de>
"""

import numpy as np
from pytest import approx, raises
from pyspherex import Expansion
import pyspherex.calculus


def test_expansion_init():
    """Test constructor"""
    exp = Expansion([[1.2], [2.3, 3.4, 4.5]])
    assert exp.coeffs[0][0] == 1.2
    assert exp.coeffs[1][0] == 2.3
    assert exp.coeffs[1][1] == 3.4
    assert exp.coeffs[1][2] == 4.5

    with raises(TypeError):
        Expansion(([1.2]))

    with raises(TypeError):
        Expansion([(1.2)])

    with raises(ValueError):
        Expansion([[1.2], [2.3, 3.4]])

    with raises(TypeError):
        Expansion([["foo"]])

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
    assert np.all(result[2][1] == \
            approx(-np.sqrt(15 / 2 / np.pi) / 2 * (np.sin(theta) * np.cos(theta))[:,None] * np.exp(1j * phi)))
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
    assert expansion.coeffs[1][2] == approx(0)

def test_expansion_from_data_y21():
    """Spherical harmonics expansion of data only containing the second degree first order sperical harmonic"""
    size_phi = 200
    size_theta = 100
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]

    data = -np.sqrt(15 / 2 / np.pi) / 2 * (np.sin(theta) * np.cos(theta))[:,None] * np.exp(1j * phi)
    expansion = Expansion.from_data(phi, theta, data, 3)

    for degree in range(len(expansion.coeffs)):
        for order, coeff in zip(range(-degree, degree + 1), expansion.coeffs[degree]):
            if degree == 2 and order == 1:
                assert coeff == approx(1, rel=1e-3)
            else:
                assert coeff == approx(0)

def test_expansion_call_sine():
    """Test `__call__` for Spherical harmonics expansion"""
    size_phi = 200
    size_theta = 100
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]

    data = np.sin(theta[:,None]) * np.sin(phi)
    expansion = Expansion.from_data(phi, theta, data, 10)

    assert np.all(data == approx(expansion(phi, theta, 10), rel=1e-1))

def test_expansion_spectrum_power():
    """Test that sum over power spectrum gives total power"""
    size_phi = 200
    size_theta = 100
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]
    degree_max = 20

    data = -4 / np.pi**2 * (np.repeat(theta[:,None], size_phi, axis=1) - np.pi / 2)**2 + 1
    expansion = Expansion.from_data(phi, theta, data, degree_max)
    power = pyspherex.calculus.sph_integrate(phi, theta, np.abs(data)**2) / 4 / np.pi

    assert np.sum(expansion.spectrum) == approx(power, rel=1e-3)
    assert expansion.power == approx(power, rel=1e-3)

def test_expansion_normalize():
    """Test normalization of spherical harmonics expansion"""
    size_phi = 200
    size_theta = 100
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]
    degree_max = 10

    coeffs = [[np.random.normal() + 1j * np.random.normal()
        for order in range(2 * degree + 1)]
        for degree in range(degree_max + 1)]

    expansion = Expansion(coeffs).normalize()
    data = expansion(phi, theta, degree_max)
    integral = pyspherex.calculus.sph_integrate(phi, theta, np.abs(data)**2)

    assert integral == approx(1, rel=1e-3)

def test_expansion_eq():
    """Test `==` operator"""
    exp1 = Expansion([[1.2], [2.3, 3.4, 4.5]])
    exp2 = Expansion([[1.2], [2.3, 3.4, 4.5]])

    assert exp1 == exp2

def test_expansion_neq():
    """Test `!=` operator"""
    exp1 = Expansion([[1.2], [2.3, 3.4, 4.5]])
    exp2 = Expansion([[1.2], [2.3, 3.4, 4.6]])

    assert exp1 != exp2

def test_expansion_add():
    """Test `+` operator"""
    exp1 = Expansion([[1], [2, 3, 4]])
    exp2 = Expansion([[5], [6, 7, 8]])
    res = Expansion([[6], [8, 10, 12]])

    assert exp1 + exp2 == res

    exp1 = Expansion([[1], [2, 3, 4]])
    exp2 = Expansion([[5]])
    res = Expansion([[6], [2, 3, 4]])

    assert exp1 + exp2 == res

def test_expansion_sub():
    """Test `-` operator"""
    exp1 = Expansion([[1], [2, 3, 4]])
    exp2 = Expansion([[5], [6, 7, 8]])
    res = Expansion([[-4], [-4, -4, -4]])

    assert exp1 - exp2 == res

    exp1 = Expansion([[1], [2, 3, 4]])
    exp2 = Expansion([[5]])
    res = Expansion([[-4], [2, 3, 4]])

    assert exp1 - exp2 == res

def test_expansion_rmul():
    """Test `*` operator"""
    exp = Expansion([[1], [2, 3, 4]])
    res = Expansion([[2], [4, 6, 8]])

    assert 2 * exp == res

def test_expansion_neg():
    """Test `*` operator"""
    exp = Expansion([[1], [2, 3, 4]])
    res = Expansion([[-1], [-2, -3, -4]])

    assert -exp == res
