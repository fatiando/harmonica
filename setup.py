"""
Build and install the project.

Uses setuptools-scm to manage version numbers using git tags.
"""
import os
from setuptools import setup, find_packages


NAME = "harmonica"
FULLNAME = "Harmonica"
AUTHOR = "The Harmonica Developers"
AUTHOR_EMAIL = "leouieda@gmail.com"
MAINTAINER = "Leonardo Uieda"
MAINTAINER_EMAIL = AUTHOR_EMAIL
LICENSE = "BSD License"
URL = "https://github.com/fatiando/harmonica"
DESCRIPTION = "Forward modeling, inversion, and processing gravity and magnetic data "
KEYWORDS = ""
with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3 :: Only",
    "License :: OSI Approved :: {}".format(LICENSE),
]
PLATFORMS = "Any"
PACKAGES = find_packages(exclude=["doc"])
SCRIPTS = []
PACKAGE_DATA = {
    "harmonica.datasets": ["registry.txt"],
    "harmonica.tests": ["data/*", "baseline/*"],
}
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "pandas",
    "numba",
    "pooch>=0.7.0",
    "xarray",
    "verde>=1.5.0",
]
PYTHON_REQUIRES = ">=3.6"

# Configuration for setupttools_scm
SETUP_REQUIRES = ["setuptools_scm"]
USE_SCM_VERSION = {
    "relative_to": __file__,
    "local_scheme": "node-and-date",
}
# Modify local_scheme with HARMONICA_VERSION_LOCAL_SCHEME env variable
# Available options:
#   - node-and-date (default)
#   - node-and-timestamp
#   - dirty-tag
#   - no-local-version (compatible with PyPI)
ENV = "HARMONICA_VERSION_LOCAL_SCHEME"
if ENV in os.environ and os.environ[ENV]:
    USE_SCM_VERSION["local_scheme"] = os.environ[ENV]


if __name__ == "__main__":
    setup(
        name=NAME,
        fullname=FULLNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        use_scm_version=USE_SCM_VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        url=URL,
        platforms=PLATFORMS,
        scripts=SCRIPTS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        python_requires=PYTHON_REQUIRES,
        setup_requires=SETUP_REQUIRES,
    )
