.. title:: Home

.. grid::
    :gutter: 2 3 3 3
    :margin: 5 5 0 0
    :padding: 0 0 0 0

    .. grid-item::
        :columns: 12 8 8 8

        .. raw:: html

            <h1 class="display-1">Harmonica</h1>

        .. div:: sd-fs-3

            Processing and modelling gravity and magnetic data

    .. grid-item::
        :columns: 12 4 4 4

        .. image:: ./_static/harmonica-logo.svg
            :width: 200px
            :class: sd-m-auto dark-light

**Harmonica** is a Python library for processing and modeling gravity and
magnetic data.

It includes common processing steps, like calculation of Bouguer and terrain
corrections, reduction to the pole, upward continuation, equivalent sources,
and more. There are forward modeling functions for basic geometric shapes, like
point sources, prisms and tesseroids. The inversion methods are implemented as
classes with an interface inspired by scikit-learn (like `Verde
<https://www.fatiando.org/verde>`__).

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


.. seealso::

    Harmonica is a part of the
    `Fatiando a Terra <https://www.fatiando.org/>`_ project.

.. admonition:: Ready for daily use but still changing
    :class: seealso

    This means that we are still adding a lot of new features and sometimes we
    make changes to the ones we already have while we try to improve the
    software based on users' experience, test new ideas, take better design
    decisions, etc.
    Some of these changes could be **backwards incompatible**. Keep that in
    mind before you update Harmonica to a newer version.

    :octicon:`comment-discussion` **We welcome any feedback and ideas!** This
    is a great time to bring new ideas on how we can improve Harmonica, feel
    free to `join the conversation <https://www.fatiando.org/contact>`__ or
    submit a new `issues on Github
    <https://github.com/fatiando/harmonica/issues>`__.

------------

Imposter syndrome disclaimer
++++++++++++++++++++++++++++

**We want your help.** No, really.

There may be a little voice inside your head that is telling you that you're
not ready to be an open source contributor; that your skills aren't nearly good
enough to contribute.
What could you possibly offer?

We assure you that the little voice in your head is wrong.

**Being a contributor doesn't just mean writing code**.
Equally important contributions include:
writing or proof-reading documentation, suggesting or implementing tests, or
even giving feedback about the project (including giving feedback about the
contribution process).
If you're coming to the project with fresh eyes, you might see the errors and
assumptions that seasoned contributors have glossed over.
If you can write any code at all, you can contribute code to open source.
We are constantly trying out new skills, making mistakes, and learning from
those mistakes.
That's how we all improve and we are happy to help others learn.

.. admonition:: How to contribute
    :class: seealso

    Please, read our `Contributor Guide
    <https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md>`_ to learn
    how you can contribute to the project.

.. note::
    *This disclaimer was adapted from the*
    `MetPy project <https://github.com/Unidata/MetPy>`__.



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

    Join the community <https://www.fatiando.org/contact>
    How to contribute <https://github.com/fatiando/harmonica/blob/main/CONTRIBUTING.md>
    Code of Conduct <https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md>
    Source code on GitHub <https://github.com/fatiando/harmonica>
    The Fatiando a Terra project <https://www.fatiando.org>
