#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long,missing-module-docstring,exec-used

import setuptools


with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='pyspherex',
    version='1.0',
    author='Martin Staab',
    author_email='martin.staab@aei.mpg.de',
    description='Python tool to perform spherical harmonics expansion.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/marstaa/PySphereX',
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'],
    keywords=['sperical harmonics', 'expansion']
)
