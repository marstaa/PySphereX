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
    exp = Expansion({0: [1.2], 1: [2.3, 3.4, 4.5]})
    assert exp.coeffs[0][0] == 1.2
    assert exp.coeffs[1][0] == 2.3
    assert exp.coeffs[1][1] == 3.4
    assert exp.coeffs[1][2] == 4.5

    with raises(TypeError):
        Expansion(([1.2]))

    with raises(TypeError):
        Expansion([(1.2)])

    with raises(ValueError):
        Expansion({0: [1.2], 1: [2.3, 3.4]})

    with raises(TypeError):
        Expansion({0: ["foo"]})

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

    for degree, orders in expansion.coeffs.items():
        for order, coeff in zip(range(-degree, degree + 1), orders):
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

    assert np.all(data == approx(expansion(phi, theta), rel=1e-1))

def test_expansion_spectrum_power():
    """Test that sum over power spectrum gives total power"""
    size_phi = 200
    size_theta = 100
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]
    degree_max = 20

    data = -4 / np.pi**2 * (np.repeat(theta[:,None], size_phi, axis=1) - np.pi / 2)**2 + 1
    expansion = Expansion.from_data(phi, theta, data, degree_max)
    degrees, spectrum = expansion.spectrum
    power = pyspherex.calculus.sph_integrate(phi, theta, np.abs(data)**2) / 4 / np.pi

    assert np.all(degrees == np.arange(degree_max + 1))
    assert np.sum(spectrum) == approx(power, rel=1e-3)
    assert expansion.power == approx(power, rel=1e-3)

def test_expansion_normalized():
    """Test normalization of spherical harmonics expansion"""
    size_phi = 200
    size_theta = 100
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]
    degree_max = 10

    coeffs = {degree: [np.random.normal() + 1j * np.random.normal()
        for order in range(2 * degree + 1)]
        for degree in range(degree_max + 1)}

    expansion = Expansion(coeffs).normalized()
    data = expansion(phi, theta)
    integral = pyspherex.calculus.sph_integrate(phi, theta, np.abs(data)**2)

    assert integral == approx(1, rel=1e-3)

def test_expansion_eq():
    """Test `==` operator"""
    exp1 = Expansion({0: [1.2], 1: [2.3, 3.4, 4.5]})
    exp2 = Expansion({0: [1.2], 1: [2.3, 3.4, 4.5]})

    assert exp1 == exp2

def test_expansion_neq():
    """Test `!=` operator"""
    exp1 = Expansion({0: [1.2], 1: [2.3, 3.4, 4.5]})
    exp2 = Expansion({0: [1.2], 1: [2.3, 3.4, 4.6]})

    assert exp1 != exp2

def test_expansion_add():
    """Test `+` operator"""
    exp1 = Expansion({0: [1], 1: [2, 3, 4]})
    exp2 = Expansion({0: [5], 1: [6, 7, 8]})
    res = Expansion({0: [6], 1: [8, 10, 12]})

    print(exp1.coeffs, exp2.coeffs, (exp1 + exp2).coeffs)

    assert exp1 + exp2 == res

    exp1 = Expansion({0: [1], 1: [2, 3, 4]})
    exp2 = Expansion({0: [5]})
    res = Expansion({0: [6], 1: [2, 3, 4]})

    assert exp1 + exp2 == res

def test_expansion_sub():
    """Test `-` operator"""
    exp1 = Expansion({0: [1], 1: [2, 3, 4]})
    exp2 = Expansion({0: [5], 1: [6, 7, 8]})
    res = Expansion({0: [-4], 1: [-4, -4, -4]})

    assert exp1 - exp2 == res

    exp1 = Expansion({0: [1], 1: [2, 3, 4]})
    exp2 = Expansion({0: [5]})
    res = Expansion({0: [-4], 1: [2, 3, 4]})

    assert exp1 - exp2 == res

def test_expansion_rmul():
    """Test `*` operator"""
    exp = Expansion({0: [1], 1: [2, 3, 4]})
    res = Expansion({0: [2], 1: [4, 6, 8]})

    assert 2 * exp == res

def test_expansion_neg():
    """Test `*` operator"""
    exp = Expansion({0: [1], 1: [2, 3, 4]})
    res = Expansion({0: [-1], 1: [-2, -3, -4]})

    assert -exp == res

def test_expansion_matmul():
    """Test overlap integral"""
    size_phi = 200
    size_theta = 100
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]
    degree_max = 10

    data1 = np.sin(theta[:,None]) * np.sin(phi)
    exp1 = Expansion.from_data(phi, theta, data1, degree_max)

    data2 = np.sin(theta[:,None]) * np.exp(1j * phi)
    exp2 = Expansion.from_data(phi, theta, data2, degree_max)
    res = -4j * np.pi / 3
    assert exp1 @ exp2 == approx(res, rel=1e-2)

def test_expansion_len():
    """Test length of expansion"""
    exp = Expansion({})
    assert len(exp) == 0

    exp = Expansion({0: [1], 1: [2, 3, 4]})
    assert len(exp) == 2

def test_expansion_slice():
    """Test slicing of expansion"""
    degree_max = 10
    coeffs = {degree: [np.random.normal() + 1j * np.random.normal()
        for order in range(2 * degree + 1)]
        for degree in range(degree_max + 1)}
    exp = Expansion(coeffs)

    sliced = exp[0]
    assert len(sliced) == 1
    assert sliced.coeffs[0] == coeffs[0]

    sliced = exp[1:5]
    assert len(sliced) == 5

    sliced = exp[1:-1]
    assert len(sliced) == degree_max

    sliced = exp[1:]
    assert len(sliced) == degree_max + 1

    sliced = exp[1:9:2]
    assert len(sliced) == 8
    assert 1 in sliced.coeffs
    assert 3 in sliced.coeffs
    assert 5 in sliced.coeffs
    assert 7 in sliced.coeffs
