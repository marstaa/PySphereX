#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines `Expansion` class.

Authors:
    Martin Staab <martin.staab@aei.mpg.de>
"""

import numpy as np
from .calculus import sph_integrate


class Expansion:
    """Defines a spherical harmonics expansion"""

    def __init__(self, coeffs):
        """Initialize a spherical harmonics expansion.

        Args:
            coeffs: coefficients of spherical harmonics expansion
        """
        self.coeffs = coeffs

    @classmethod
    def from_data(cls, phi, theta, data, degree_max):
        """Perform spherical harmonics expansion of discrete data.

        Args:
            phi: array of azimuthal angles
            theta: array of polar angles
            data: 2-dim array of data
            degree_max: maximum degree
        """
        coeffs = []
        basis = cls.generate_sph_basis(phi, theta, degree_max)

        for degree in range(degree_max + 1):
            coeffs.append([])
            for order in range(-degree, degree + 1):
                if order < 0:
                    coeff = sph_integrate(phi, theta, data * (-1)**order * basis[degree][-order])
                elif order >= 0:
                    coeff = sph_integrate(phi, theta, data * np.conj(basis[degree][order]))
                coeffs[degree].append(coeff)

        return cls(coeffs)

    @staticmethod
    def generate_sph_basis(phi, theta, degree_max):
        """Generate spherical harmonics basis.

        Args:
            phi: array of azimuthal angles
            theta: array of polar angles
            degree_max: maximum degree
        """
        legendre = [[np.array([1])]]
        harmonics = [[np.array([np.sqrt(1 / 4 / np.pi)])]]

        x = np.cos(theta)
        y = np.sin(theta)

        for degree in range(1, degree_max + 1):
            legendre.append([])
            harmonics.append([])
            for order in range(degree + 1):
                if order == degree - 1:
                    legendre[degree].append(x * (2 * degree - 1) * legendre[degree-1][degree-1])
                elif order == degree:
                    legendre[degree].append(-y * (2 * degree - 1) * legendre[degree-1][degree-1])
                else:
                    tmp = (2 * degree - 1) * x * legendre[degree-1][order]
                    tmp -= (degree + order - 1) * legendre[degree-2][order]
                    legendre[degree].append(tmp / (degree - order))

                tmp = np.sqrt((2 * degree + 1) / 4 / np.pi \
                        * np.math.factorial(degree - order) / np.math.factorial(degree + order))
                if order == 0:
                    harmonics[degree].append(tmp * legendre[degree][order][:,None])
                else:
                    harmonics[degree].append(tmp * legendre[degree][order][:,None] * np.exp(1j * order * phi))

        return harmonics

    def __call__(self, phi, theta, degree_max):
        """Evaluate spherical harmonics expansion at given points.

        Args:
            phi: array of azimuthal angles
            theta: array of polar angles
            degree_max: maximum degree
        """

        if degree_max > len(self.coeffs) - 1:
            raise ValueError(f'degree_max ({degree_max}) must not exceed the degree '
                              'of the expansion ({len(self.coeffs) - 1})')

        basis = self.generate_sph_basis(phi, theta, degree_max)
        result = np.zeros((theta.size, phi.size), dtype=np.complex128)

        for degree in range(degree_max + 1):
            for order, coeff in zip(range(-degree, degree + 1), self.coeffs[degree]):
                if order < 0:
                    result += coeff * (-1)**order * np.conj(basis[degree][-order])
                elif order >= 0:
                    result += coeff * basis[degree][order]

        return result
