#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests calculus routines.

Authors:
    Martin Staab <martin.staab@aei.mpg.de>
"""

import numpy as np
from pytest import approx
import pyspherex.calculus


def test_sph_integrate_ones():
    """Integrate over unit sphere of constant value."""
    size_phi = 2000
    size_theta = 1000
    phi = np.arange(size_phi) * 2 * np.pi / size_phi
    theta = np.linspace(0, np.pi, size_theta + 2)[1:-1]

    data = np.ones((size_theta, size_phi))
    result = pyspherex.calculus.sph_integrate(phi, theta, data)
    expected = 4 * np.pi
    assert result == approx(expected)
