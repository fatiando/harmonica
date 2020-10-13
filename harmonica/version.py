# pylint: disable=invalid-name
"""
Get the version number and commit hash from setuptools_scm.
"""
from pkg_resources import get_distribution

# Append a "v" before the version returned by setuptools_scm so it can look
# like: v0.1.0
full_version = "v" + get_distribution(__name__).version
