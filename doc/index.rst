.. title:: Home

========
|banner|
========

.. |banner| image:: _static/readme-banner.png
    :alt: Harmonica Documentation
    :align: middle

**Harmonica** is a Python library for processing and modeling gravity and
magnetic data.

It includes common processing steps, like calculation of Bouguer and terrain
corrections, reduction to the pole, upward continuation, equivalent sources,
and more. There are forward modeling functions for basic geometric shapes, like
point sources, prisms and tesseroids. The inversion methods are implemented as
classes with an interface inspired by scikit-learn (like `Verde
<https://www.fatiando.org/verde>`__).

.. admonition:: Ready for daily use but still changing
    :class: seealso

    This means that we are still adding a lot of new features and sometimes we
    make changes to the ones we already have while we try to improve the
    software based on users' experience, test new ideas, take better design
    decisions, etc.
    Some of these changes could be **backwards incompatible**. Keep that in
    mind before you update Harmonica to a newer version.

    **We welcome any feedback and ideas!** This is a great time to bring new
    ideas on how we can improve Harmonica, feel free to `join the
    conversation <https://www.fatiando.org/contact>`__ or submit a new
    `issues on Github <https://github.com/fatiando/harmonica/issues>`__.

.. panels::
    :header: text-center text-large
    :card: border-1 m-1 text-center

    **Getting started**
    ^^^^^^^^^^^^^^^^^^^

    New to Harmonica? Start here!

    .. link-button:: overview
        :type: ref
        :text: Overview
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Need help?**
    ^^^^^^^^^^^^^^

    Ask on our community channels

    .. link-button:: https://www.fatiando.org/contact
        :type: url
        :text: Join the conversation
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Reference documentation**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    A list of modules and functions

    .. link-button:: api
        :type: ref
        :text: API reference
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Using Harmonica for research?**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Citations help support our work

    .. link-button:: citing
        :type: ref
        :text: Cite Harmonica
        :classes: btn-outline-primary btn-block stretched-link

.. seealso::

    Harmonica is a part of the
    `Fatiando a Terra <https://www.fatiando.org/>`_ project.

----

Table of contents
-----------------

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    overview.rst
    install.rst
    gallery/index.rst


.. toctree::
    :maxdepth: 2
    :caption: User Guide
    :includehidden:

    user_guide/coordinate_systems.rst
    user_guide/forward_modelling/index.rst
    user_guide/gravity_disturbance.rst
    user_guide/topographic_correction.rst
    user_guide/equivalent_sources/index.rst

.. toctree::
    :maxdepth: 1
    :caption: Reference documentation

    api/index.rst
    citing.rst
    changes.rst
    references.rst
    versions.rst

.. toctree::
    :maxdepth: 1
    :caption: Community

    Join the community <http://contact.fatiando.org>
    How to contribute <https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md>
    Code of Conduct <https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md>
    Source code on GitHub <https://github.com/fatiando/harmonica>
    The Fatiando a Terra project <https://www.fatiando.org>
