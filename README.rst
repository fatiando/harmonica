.. image:: https://github.com/fatiando/harmonica/raw/master/doc/_static/readme-banner.png
    :alt: Harmonica

`Documentation <http://www.fatiando.org/harmonica>`__ |
`Documentation (dev version) <http://www.fatiando.org/harmonica/dev>`__ |
`Contact <https://gitter.im/fatiando/fatiando>`__ |
Part of the `Fatiando a Terra <https://www.fatiando.org>`__ project

.. image:: http://img.shields.io/pypi/v/harmonica.svg?style=flat-square
    :alt: Latest version on PyPI
    :target: https://pypi.python.org/pypi/harmonica
.. image:: http://img.shields.io/travis/fatiando/harmonica/master.svg?style=flat-square&label=Linux|Mac
    :alt: TravisCI build status
    :target: https://travis-ci.org/fatiando/harmonica
.. image:: http://img.shields.io/appveyor/ci/fatiando/harmonica/master.svg?style=flat-square&label=Windows
    :alt: AppVeyor build status
    :target: https://ci.appveyor.com/project/fatiando/harmonica
.. image:: https://img.shields.io/codecov/c/github/fatiando/harmonica/master.svg?style=flat-square
    :alt: Test coverage status
    :target: https://codecov.io/gh/fatiando/harmonica
.. image:: https://img.shields.io/codeclimate/maintainability/fatiando/harmonica.svg?style=flat-square
    :alt: Code quality status
    :target: https://codeclimate.com/github/fatiando/harmonica
.. image:: https://img.shields.io/codacy/grade/6b698defc0df47288a634930d41a9d65.svg?style=flat-square&label=codacy
    :alt: Code quality grade on codacy
    :target: https://www.codacy.com/app/leouieda/harmonica
.. image:: https://img.shields.io/pypi/pyversions/harmonica.svg?style=flat-square
    :alt: Compatible Python versions.
    :target: https://pypi.python.org/pypi/harmonica
.. image:: https://img.shields.io/gitter/room/fatiando/fatiando.svg?style=flat-square
    :alt: Chat room on Gitter
    :target: https://gitter.im/fatiando/fatiando


.. placeholder-for-doc-index


Disclaimer
----------

🚨 **This package is in early stages of design and implementation.** 🚨

We welcome any feedback and ideas!
Let us know by submitting
`issues on Github <https://github.com/fatiando/harmonica/issues>`__
or send us a message on our
`Gitter chatroom <https://gitter.im/fatiando/fatiando>`__.


About
-----

*Harmonica* is a Python library for processing and modeling gravity and magnetic data.
It includes common processing steps, like calculation of normal gravity, terrain
correction, reduction to the pole, upward continuation, equivalent layers, and more.
There are forward modeling functions for basic geometric shapes, like spheres, prisms,
polygonal prisms, and tesseroids. The inversion methods are implemented as classes with
an interface inspired by scikit-learn (like `Verde <https://www.fatiando.org/verde>`__).


Project goals
-------------

These are the long-term goals for Harmonica:

* Efficient, well designed, and fully tested code for gravity and magnetic data.
* Cover the entire data life-cycle: from raw data to 3D Earth model.
* Focus on best-practices to discourage misuse of methods, particularly inversion.
* Easily extended code to enable research on the development of new methods.

See the `Github milestones <https://github.com/fatiando/harmonica/milestones>`__ for
short-term goals.

Things that will *not* be covered in Harmonica:

* Multi-physics partial differential equation solvers. Use
  `SimPEG <http://www.simpeg.xyz/>`__ or `PyGIMLi <https://www.pygimli.org/>`__ instead.
* Generic grid processing methods (like horizontal derivatives and FFT). These should be
  implemented in `Verde <https://www.fatiando.org/verde>`__.
* Data visualization.
* GUI applications.


Contacting Us
-------------

* Most discussion happens `on Github <https://github.com/fatiando/harmonica>`__.
  Feel free to `open an issue
  <https://github.com/fatiando/harmonica/issues/new>`__ or comment
  on any open issue or pull request.
* We have `chat room on Gitter <https://gitter.im/fatiando/fatiando>`__
  where you can ask questions and leave comments.


Contributing
------------

Code of conduct
+++++++++++++++

Please note that this project is released with a
`Contributor Code of Conduct <https://github.com/fatiando/harmonica/blob/master/CODE_OF_CONDUCT.md>`__.
By participating in this project you agree to abide by its terms.

Contributing Guidelines
+++++++++++++++++++++++

Please read our
`Contributing Guide <https://github.com/fatiando/harmonica/blob/master/CONTRIBUTING.md>`__
to see how you can help and give feedback.

Imposter syndrome disclaimer
++++++++++++++++++++++++++++

**We want your help.** No, really.

There may be a little voice inside your head that is telling you that you're
not ready to be an open source contributor; that your skills aren't nearly good
enough to contribute.
What could you possibly offer?

We assure you that the little voice in your head is wrong.

**Being a contributor doesn't just mean writing code**.
Equality important contributions include:
writing or proof-reading documentation, suggesting or implementing tests, or
even giving feedback about the project (including giving feedback about the
contribution process).
If you're coming to the project with fresh eyes, you might see the errors and
assumptions that seasoned contributors have glossed over.
If you can write any code at all, you can contribute code to open source.
We are constantly trying out new skills, making mistakes, and learning from
those mistakes.
That's how we all improve and we are happy to help others learn.

*This disclaimer was adapted from the*
`MetPy project <https://github.com/Unidata/MetPy>`__.


License
-------

This is free software: you can redistribute it and/or modify it under the terms
of the **BSD 3-clause License**. A copy of this license is provided in
`LICENSE.txt <https://github.com/fatiando/harmonica/blob/master/LICENSE.txt>`__.


Documentation for other versions
--------------------------------

* `Development <http://www.fatiando.org/harmonica/dev>`__ (reflects the *master* branch on
  Github)
* `Latest release <http://www.fatiando.org/harmonica/latest>`__
