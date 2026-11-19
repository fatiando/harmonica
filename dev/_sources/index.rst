.. title:: Home

Harmonica
=========

.. div:: sd-fs-3

    Processing and modelling gravity and magnetic data.


**Harmonica** is a Python library for processing and modeling gravity and
magnetic data.
It includes common processing steps, like calculation of Bouguer and terrain
corrections, reduction to the pole, upward continuation, equivalent sources,
and more. There are forward modeling functions for basic geometric shapes, like
point sources, prisms and tesseroids. The inversion methods are implemented as
classes with an interface inspired by scikit-learn (like `Verde
<https://www.fatiando.org/verde>`__).

.. seealso::

    Harmonica is a part of the
    `Fatiando a Terra <https://www.fatiando.org/>`_ project.

.. grid:: 1 2 1 2
    :margin: 5 5 0 0
    :padding: 0 0 0 0
    :gutter: 4

    .. grid-item-card:: :octicon:`rocket` Getting started
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        New to Harmonica? Start here!

        .. button-ref:: overview
            :click-parent:
            :color: primary
            :outline:
            :expand:

    .. grid-item-card:: :octicon:`comment-discussion` Need help?
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        Ask on our community channels

        .. button-link:: https://www.fatiando.org/contact
            :click-parent:
            :color: primary
            :outline:
            :expand:

             Join the conversation

    .. grid-item-card:: :octicon:`file-badge` Reference documentation
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        A list of modules and functions

        .. button-ref:: api
            :click-parent:
            :color: primary
            :outline:
            :expand:


    .. grid-item-card:: :octicon:`bookmark` Using Harmonica for research?
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        Citations help support our work

        .. button-ref:: citing
            :click-parent:
            :color: primary
            :outline:
            :expand:

----


Project status
--------------

**Harmonica is ready for use but still changing.** This means that we
sometimes break backwards compatibility as we try to improve the
software based on user experience, new ideas, better design decisions,
etc. Please keep that in mind before you update Harmonica to a newer
version.

**We welcome feedback and ideas!** This is a great time to bring new
ideas on how we can improve the project. `Join the
conversation <https://www.fatiando.org/contact>`__ or submit `issues on
GitHub <https://github.com/fatiando/harmonica/issues>`__.


Getting involved
----------------

🗨️ **Contact us:** Find out more about how to reach us at
`fatiando.org/contact <https://www.fatiando.org/contact/>`__.

👩🏾‍💻 **Contributing to project development:** Please read our
`Contributing
Guide <https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md>`__
to see how you can help and give feedback.

🤖 **Policy on AI usage:** Please read our `AI Usage
Policy <https://github.com/fatiando/community/blob/main/AI.md>`__.

🧑🏾‍🤝‍🧑🏼 **Code of conduct:** This project is released with a `Code of
Conduct <https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md>`__.
By participating in this project you agree to abide by its terms.

   **Imposter syndrome disclaimer:** We want your help. **No, really.**
   There may be a little voice inside your head that is telling you that
   you’re not ready, that you aren’t skilled enough to contribute. We
   assure you that the little voice in your head is wrong. Most
   importantly, **there are many valuable ways to contribute besides
   writing code**.

   *This disclaimer was adapted from the* `MetPy
   project <https://github.com/Unidata/MetPy>`__.

License
-------

This is free software: you can redistribute it and/or modify it under
the terms of the **BSD 3-clause License**. A copy of this license is
provided in
`LICENSE.txt <https://github.com/fatiando/harmonica/blob/main/LICENSE.txt>`__.


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Getting Started

    overview.rst
    install.rst
    gallery/index.rst


.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: User Guide
    :includehidden:

    user_guide/coordinate_systems.rst
    user_guide/forward_modelling/index.rst
    user_guide/gravity_disturbance.rst
    user_guide/topographic_correction.rst
    user_guide/equivalent_sources/index.rst
    user_guide/transformations.rst
    user_guide/igrf.rst


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Reference documentation

    api/index.rst
    citing.rst
    references.rst
    changes.rst
    compatibility.rst
    versions.rst

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Community

    Join our community <https://www.fatiando.org/contact>
    How to contribute <https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md>
    Code of Conduct <https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md>
    Source code on GitHub <https://github.com/fatiando/harmonica>
    The Fatiando a Terra project <https://www.fatiando.org>

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Code and packages

    GitHub <https://github.com/fatiando/harmonica>
    PyPI <https://pypi.python.org/pypi/harmonica>
    conda-forge <https://github.com/conda-forge/harmonica-feedstock>
    Zenodo <https://doi.org/10.5281/zenodo.7851747>
