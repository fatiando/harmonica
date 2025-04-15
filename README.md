<img src="https://github.com/fatiando/harmonica/raw/main/doc/_static/readme-banner.png" alt="Harmonica">

<h2 align="center">Processing and modelling gravity and magnetic data</h2>

<p align="center">
<a href="https://www.fatiando.org/harmonica"><strong>Documentation</strong> (latest)</a> •
<a href="https://www.fatiando.org/harmonica/dev"><strong>Documentation</strong> (main branch)</a> •
<a href="https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md"><strong>Contributing</strong></a> •
<a href="https://www.fatiando.org/contact/"><strong>Contact</strong></a> •
<a href="https://github.com/orgs/fatiando/discussions"><strong>Ask a question</strong></a>
</p>

<p align="center">
Part of the <a href="https://www.fatiando.org"><strong>Fatiando a Terra</strong></a> project
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/harmonica"><img src="http://img.shields.io/pypi/v/harmonica.svg?style=flat-square" alt="Latest version on PyPI"/></a>
<a href="https://github.com/conda-forge/harmonica-feedstock"><img src="https://img.shields.io/conda/vn/conda-forge/harmonica.svg?style=flat-square" alt="Latest version on conda-forge"/></a>
<a href="https://pypi.python.org/pypi/harmonica"><img src="https://img.shields.io/pypi/pyversions/harmonica.svg?style=flat-square" alt="Compatible Python versions."/></a>
<a href="https://doi.org/10.5281/zenodo.3628741"><img src="https://img.shields.io/badge/doi-10.5281%2Fzenodo.3628741-blue.svg?style=flat-square" alt="Digital Object Identifier for the Zenodo archive"/></a>
</p>

# About

*Harmonica* is a Python library for processing and modeling gravity and
magnetic data. It includes common processing steps, like calculation of Bouguer
and terrain corrections, reduction to the pole, upward continuation, equivalent
sources, and more. There are forward modeling functions for basic geometric
shapes, like point sources, prisms and tesseroids. The inversion methods are
implemented as classes with an interface inspired by scikit-learn (like
[Verde](https://www.fatiando.org/verde)).

## Project goals

These are the long-term goals for Harmonica:

- Efficient, well designed, and fully tested code for gravity and
  magnetic data.
- Cover the entire data life-cycle: from raw data to 3D Earth model.
- Focus on best-practices to discourage misuse of methods,
  particularly inversion.
- Easily extended code to enable research on the development of new
  methods.

See the [GitHub milestones](https://github.com/fatiando/harmonica/milestones)
for short-term goals.

Things that will *not* be covered in Harmonica:

- Multi-physics partial differential equation solvers. Use
  [SimPEG](http://www.simpeg.xyz/) or [PyGIMLi](https://www.pygimli.org/)
  instead.
- Generic grid processing methods (like FFT and standards interpolation).
  We'll rely on [Verde](https://www.fatiando.org/verde),
  [xrft](https://xrft.readthedocs.io/en/latest/) and
  [xarray](https://xarray.dev) for those.
- Data visualization.
- GUI applications.

## Project status

🚨 **Harmonica is in early stages of design and implementation.** 🚨

We welcome any feedback and ideas! Let us know by submitting
[issues on GitHub](https://github.com/fatiando/harmonica/issues) or
[joining our community](https://www.fatiando.org/contact).

## Getting involved

🗨️ **Contact us:**
Find out more about how to reach us at
[fatiando.org/contact](https://www.fatiando.org/contact/).

👩🏾‍💻 **Contributing to project development:**
Please read our
[Contributing Guide](https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md)
to see how you can help and give feedback.

🧑🏾‍🤝‍🧑🏼 **Code of conduct:**
This project is released with a
[Code of Conduct](https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

> **Imposter syndrome disclaimer:**
> We want your help. **No, really.** There may be a little voice inside your
> head that is telling you that you're not ready, that you aren't skilled
> enough to contribute. We assure you that the little voice in your head is
> wrong. Most importantly, **there are many valuable ways to contribute besides
> writing code**.
>
> *This disclaimer was adapted from the*
> [MetPy project](https://github.com/Unidata/MetPy).

# License

This is free software: you can redistribute it and/or modify it under the terms
of the **BSD 3-clause License**. A copy of this license is provided in
[`LICENSE.txt`](https://github.com/fatiando/harmonica/blob/main/LICENSE.txt).
