# pylint: disable=invalid-name
"""
Get the version number and commit hash from setuptools-scm.
"""
from pkg_resources import get_distribution

# Append a "v" before the version returned by setuptools-scm so it can look
# like: v0.1.0
full_version = "v" + get_distribution("harmonica").version
