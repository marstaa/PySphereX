# PySphereX

![zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.5520636.svg)

PySphereX is a Python tool to perform spherical harmonics expansion of data given on a uniformly spaced grid on a sphere. Its features include:
 * `Expansion` class that defines an arbitrary complex spherical harmonics expansion
 * Convenience constructor `Expansion.from_data` that initializes `Expansion` object from gridded data on a sphere up to a maximum degree
 * Evaluation of `Expansion` object at arbitrary coordinates
 * Angular power spectrum
 * Algebraic operations: Addition, subtraction and overlap between two `Expansion` objects

## Getting Started

Make sure that Python 3.7 or newer is available and install `pyspherex` using [pip](https://pypi.org/project/pip/),
```
pip install pyspherex
```
As a "getting started guide" we provide an example (in folder `examples/`) that shows the basic functionalities of this package. For further information see the doc-strings and the unit tests.

## Contributing

### Report an issue

We use the issue-tracking management system associated with the project provided by Github. If you want to report a bug or request a feature, open an issue at [https://github.com/marstaa/PySphereX/issues](https://github.com/marstaa/PySphereX/issues). You may also comment on existing issues.

### Development environment

We strongly recommend to use [Python virtual environments](https://docs.python.org/3/tutorial/venv.html).

To setup the development environment, use the following commands:
```
git clone git@github.com:marstaa/PySphereX.git
cd PySphereX
python -m venv venv
source ./venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Workflow

The project's development workflow is based on the issue-tracking system provided by Github, as well as peer-reviewed pull requests. This ensures high-quality standards.

Issues are solved by creating branches and opening pull requests. Only the assignee of the related issue and pull request can push commits on the branch. Once all the changes have been pushed, the pull request can be marked as ready for review and is assigned to a reviewer. They can push new changes to the branch, or request changes to the original author by re-assigning the pull request to them. When the pull request is accepted, the branch is merged onto main, deleted, and the associated issue is closed.

### Pylint and pytest

We enforce [PEP 8 (Style Guide for Python Code)](https://www.python.org/dev/peps/pep-0008/) with [Pylint](http://pylint.pycqa.org/) syntax checking, and testing of the code via unit and integration tests with the [pytest](https://docs.pytest.org/) framework. Both are implemented in the continuous integration system. Only if all tests pass successfully a pull request can be merged.

You can run them locally
```
pylint */**.py
pytest
```

## How to cite
If you intend to use this package in your own research please be sure to cite it. For BibTex we suggest:
```
@misc{pyspherex,
    author = {Martin Staab},
    title = {PySphereX},
    howpublished = {\url{http://github.com/marstaa/PySphereX}},
    year = {2021},
    DOI={<insert DOI from Zenodo>}
}
```

## Acknowledgements
Many thanks to Ulrike Proske (prosku) for providing the example script.

## Contact

* Martin Staab (martin.staab@aei.mpg.de)
