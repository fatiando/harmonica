

.. _sphx_glr_gallery_equivalent_sources:

Equivalent Sources
------------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="By default, the harmonica.EquivalentSources class locates one point source beneath each data point during the fitting process. Alternatively, we can use another strategy: the block-averaged sources, introduced in [Soler2021]_.">

.. only:: html

  .. image:: /gallery/equivalent_sources/images/thumb/sphx_glr_block_averaged_sources_thumb.png
    :alt:

  :doc:`/gallery/equivalent_sources/block_averaged_sources`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gridding with block-averaged equivalent sources</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Most potential field surveys gather data along scattered and uneven flight lines or ground measurements. For a great number of applications we may need to interpolate these data points onto a regular grid at a constant altitude. Upward-continuation is also a routine task for smoothing, noise attenuation, source separation, etc.">

.. only:: html

  .. image:: /gallery/equivalent_sources/images/thumb/sphx_glr_cartesian_thumb.png
    :alt:

  :doc:`/gallery/equivalent_sources/cartesian`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gridding and upward continuation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When trying to grid a very large dataset, the regular harmonica.EquivalentSources might not be the best option: they will require a lot of memory for storing the Jacobian matrices involved in the fitting process of the source coefficients. Instead, we can make use of the gradient-boosted equivalent sources, introduced in [Soler2021]_ and available in Harmonica through the harmonica.EquivalentSourcesGB class. The gradient-boosted equivalent sources divide the survey region in overlapping windows of equal size and fit the source coefficients iteratively, considering the sources and data points that fall under each window at a time. The order in which the windows are visited is randomized to improve convergence of the algorithm.">

.. only:: html

  .. image:: /gallery/equivalent_sources/images/thumb/sphx_glr_gradient_boosted_thumb.png
    :alt:

  :doc:`/gallery/equivalent_sources/gradient_boosted`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient-boosted equivalent sources</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The curvature of the Earth must be taken into account when gridding and processing magnetic or gravity data on large regions. In these cases, projecting the data may introduce errors due to the distortions caused by the projection.">

.. only:: html

  .. image:: /gallery/equivalent_sources/images/thumb/sphx_glr_spherical_thumb.png
    :alt:

  :doc:`/gallery/equivalent_sources/spherical`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gridding in spherical coordinates</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /gallery/equivalent_sources/block_averaged_sources
   /gallery/equivalent_sources/cartesian
   /gallery/equivalent_sources/gradient_boosted
   /gallery/equivalent_sources/spherical

