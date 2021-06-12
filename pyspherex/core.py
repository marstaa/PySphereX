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
        if not isinstance(coeffs, dict):
            raise TypeError('`coeffs` must be a dict')

        for degree, orders in coeffs.items():
            if not isinstance(degree, int):
                raise TypeError('keys of `coeffs` must be integers')
            if not isinstance(orders, list):
                raise TypeError('`coeffs` must only contain lists')
            if len(orders) != 2 * degree + 1:
                raise ValueError(f'`coeffs[{degree}]` must be of length {2 * degree + 1}')
            for coeff in orders:
                if not isinstance(coeff, (int, float, complex)):
                    raise TypeError(f'`coeffs[{degree}] must only contain integers or complex floats`')
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
        coeffs = {}
        basis = cls.generate_sph_basis(phi, theta, degree_max)

        for degree in range(degree_max + 1):
            coeffs[degree] = []
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

    def __call__(self, phi, theta):
        """Evaluate spherical harmonics expansion at given points.

        Args:
            phi: array of azimuthal angles
            theta: array of polar angles
            degree_max: maximum degree
        """
        basis = self.generate_sph_basis(phi, theta, max(self.coeffs.keys()))
        result = np.zeros((theta.size, phi.size), dtype=np.complex128)

        for degree, orders in self.coeffs.items():
            for order, coeff in zip(range(-degree, degree + 1), orders):
                if order < 0:
                    result += coeff * (-1)**order * np.conj(basis[degree][-order])
                elif order >= 0:
                    result += coeff * basis[degree][order]

        return result

    @property
    def spectrum(self):
        """Calculate power spectrum."""
        degrees = np.array(list(self.coeffs.keys()))
        powers = np.array([np.sum(np.abs(orders)**2) / 4 / np.pi
            for orders in self.coeffs.values()])
        return degrees, powers

    @property
    def power(self):
        """Calculate total power"""
        return np.sum(self.spectrum[1])

    def normalized(self):
        """Normalize spherical harmonics expansion"""
        factor = np.sqrt(self.power * 4 * np.pi)
        coeffs_norm = {degree: [coeff / factor for coeff in orders] for degree, orders in self.coeffs.items()}
        return Expansion(coeffs_norm)

    def __eq__(self, other):
        return self.coeffs == other.coeffs

    def __neq__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        coeffs_new = {}
        for coeffs in [self.coeffs, other.coeffs]:
            for degree, orders in coeffs.items():
                if degree in coeffs_new:
                    coeffs_new[degree] = [sum(lists) for lists in zip(coeffs_new[degree], orders)]
                else:
                    coeffs_new[degree] = orders.copy()

        return Expansion(coeffs_new)

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, factor):
        coeffs_new = {degree: [factor * coeff for coeff in orders] for degree, orders in self.coeffs.items()}
        return Expansion(coeffs_new)

    def __neg__(self):
        return -1 * self

    def __matmul__(self, other):
        """Calculate overlap integral between two expansions"""
        res = 0
        degrees = set(self.coeffs.keys()) & set(other.coeffs.keys())
        for degree in degrees:
            res += sum(coeff1 * coeff2.conjugate()
                for coeff1, coeff2 in zip(self.coeffs[degree], other.coeffs[degree]))
        return res

    def __len__(self):
        if self.coeffs:
            return max(self.coeffs.keys()) + 1
        return 0

    def __getitem__(self, key):
        if isinstance(key, list):
            coeffs_new = {}
            for degree in key:
                if degree in self.coeffs:
                    coeffs_new[degree] = self.coeffs[degree].copy()
            return Expansion(coeffs_new)
        if isinstance(key, int):
            return self[[key]]
        if isinstance(key, slice):
            return self[list(range(*key.indices(len(self.coeffs))))]
        raise TypeError('`key` must be an index, a slice or a list of integers')
