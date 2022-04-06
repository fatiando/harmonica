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

.. admonition:: This software is in the early stages of design and implementation
    :class: attention

    This means that will occasionally make **backwards incompatible** changes
    as we try to improve the software, test new ideas, and settle on the
    project scope.
    **We welcome any feedback and ideas!** Let us know by submitting
    `issues on Github <https://github.com/fatiando/harmonica/issues>`__
    or send us a message on our
    `Slack chatroom <http://contact.fatiando.org>`__.

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
    :maxdepth: 1
    :caption: User Guide

    user_guide/coordinate_systems.rst
    user_guide/forward_modelling/index.rst
    user_guide/gravity_processing/index.rst
    user_guide/equivalent_sources/index.rst
    user_guide/read_icgem.py

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
