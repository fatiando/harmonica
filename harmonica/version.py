# pylint: disable=invalid-name
"""
Get the version number and commit hash from setuptools-scm.
"""
from pkg_resources import get_distribution


# Get semantic version through setuptools-scm
version = get_distribution("harmonica").version
# Append a "v" before the semver version so it looks like: v0.1.0
full_version = "v" + version
