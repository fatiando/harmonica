<img src="https://github.com/fatiando/harmonica/raw/main/doc/_static/readme-banner.png" alt="Harmonica">

<h1 align="center">Harmonica</h1>

<h2 align="center">Processing and modelling gravity and magnetic data</h2>

<p align="center">
<a href="https://www.fatiando.org/harmonica"><strong>Documentation</strong> (latest)</a> •
<a href="https://www.fatiando.org/harmonica/dev"><strong>Documentation</strong> (main branch)</a> •
<a href="https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md"><strong>Contributing</strong></a> •
<a href="https://www.fatiando.org/contact/"><strong>Contact</strong></a>
</p>

<p align="center">
Part of the <a href="https://www.fatiando.org"><strong>Fatiando a Terra</strong></a> project
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/harmonica">
    <img
        src="http://img.shields.io/pypi/v/harmonica.svg?style=flat-square"
        alt="Latest version on PyPI"
    />
</a>
<a href="https://github.com/fatiando/harmonica/actions">
    <img
        src="https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Ffatiando%2Fharmonica%2Fbadge%3Fref%3Dmain&amp;style=flat-square&amp;logo=none"
        alt="GitHub Actions status"
    />
</a>
<a href="https://codecov.io/gh/fatiando/harmonica">
    <img
        src="https://img.shields.io/codecov/c/github/fatiando/harmonica/main.svg?style=flat-square"
        alt="Test coverage status"
    />
</a>
<a href="https://pypi.python.org/pypi/harmonica">
    <img
        src="https://img.shields.io/pypi/pyversions/harmonica.svg?style=flat-square"
        alt="Compatible Python versions."
    />
</a>
<a href="https://doi.org/10.5281/zenodo.3628741">
    <img
    src="https://img.shields.io/badge/doi-10.5281%2Fzenodo.3628741-blue.svg?style=flat-square"
    alt="Digital Object Identifier for the Zenodo archive"
    />
</a>
</p>


# Disclaimer

🚨 **This package is in early stages of design and implementation.** 🚨

We welcome any feedback and ideas! Let us know by submitting [issues on
Github](https://github.com/fatiando/harmonica/issues) or send us a message on
our [Slack chatroom](http://contact.fatiando.org).

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

See the [Github milestones](https://github.com/fatiando/harmonica/milestones)
for short-term goals.

Things that will *not* be covered in Harmonica:

- Multi-physics partial differential equation solvers. Use
  [SimPEG](http://www.simpeg.xyz/) or [PyGIMLi](https://www.pygimli.org/)
  instead.
- Generic grid processing methods (like horizontal derivatives and FFT). These
  should be implemented in [Verde](https://www.fatiando.org/verde).
- Data visualization.
- GUI applications.

# Contacting Us

- Most discussion happens [on Github](https://github.com/fatiando/harmonica).
  Feel free to [open an
  issue](https://github.com/fatiando/harmonica/issues/new) or comment on any
  open issue or pull request.
- We have [chat room on Slack](http://contact.fatiando.org) where you can ask
  questions and leave comments.

# Contributing

## Code of conduct

Please note that this project is released with a [Contributor Code of
Conduct](https://github.com/fatiando/harmonica/blob/main/CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

## Contributing Guidelines

Please read our [Contributing
Guide](https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md) to see
how you can help and give feedback.

## Imposter syndrome disclaimer

**We want your help.** No, really.

There may be a little voice inside your head that is telling you that
you're not ready to be an open source contributor; that your skills
aren't nearly good enough to contribute. What could you possibly offer?

We assure you that the little voice in your head is wrong.

**Being a contributor doesn't just mean writing code**. Equally
important contributions include: writing or proof-reading documentation,
suggesting or implementing tests, or even giving feedback about the
project (including giving feedback about the contribution process). If
you're coming to the project with fresh eyes, you might see the errors
and assumptions that seasoned contributors have glossed over. If you can
write any code at all, you can contribute code to open source. We are
constantly trying out new skills, making mistakes, and learning from
those mistakes. That's how we all improve and we are happy to help
others learn.

*This disclaimer was adapted from the* [MetPy
project](https://github.com/Unidata/MetPy).

# License

This is free software: you can redistribute it and/or modify it under the terms
of the **BSD 3-clause License**. A copy of this license is provided in
[LICENSE.txt](https://github.com/fatiando/harmonica/blob/main/LICENSE.txt).

# Documentation for other versions

- [Development](http://www.fatiando.org/harmonica/dev) (reflects the *main*
  branch on Github)
- [Latest release](http://www.fatiando.org/harmonica/latest)
- [v0.4.0](http://www.fatiando.org/harmonica/v0.4.0)
- [v0.3.3](http://www.fatiando.org/harmonica/v0.3.3)
- [v0.3.2](http://www.fatiando.org/harmonica/v0.3.2)
- [v0.3.1](http://www.fatiando.org/harmonica/v0.3.1)
- [v0.3.0](http://www.fatiando.org/harmonica/v0.3.0)
- [v0.2.1](http://www.fatiando.org/harmonica/v0.2.1)
- [v0.2.0](http://www.fatiando.org/harmonica/v0.2.0)
- [v0.1.0](http://www.fatiando.org/harmonica/v0.1.0)
