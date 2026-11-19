:orphan:

.. _gallery:

Gallery
=======

This gallery contains a selection examples of what Harmonica can do.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Gravity disturbances are the differences between the measured gravity and a reference (normal) gravity produced by an ellipsoid. The disturbances are what allows geoscientists to infer the internal structure of the Earth. We&#x27;ll use the boule.Ellipsoid.normal_gravity function from boule to calculate the global gravity disturbance of the Earth using our sample gravity data.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_gravity_disturbance_thumb.png
    :alt:

  :doc:`/gallery/gravity_disturbance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gravity Disturbances</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The gravity disturbance is the observed gravity minus the normal gravity (boule.Ellipsoid.normal_gravity). The signal that is left is assumed to be due to the differences between the actual Earth and the reference ellipsoid. Big components in this signal are the effects of topographic masses outside of the ellipsoid and residual effects of the oceans (we removed ellipsoid crust where there was actually ocean water). These two components are relatively well known and can be removed from the gravity disturbance. The simplest way of calculating the effects of topography and oceans is through the Bouguer plate approximation.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_gravity_disturbance_topofree_thumb.png
    :alt:

  :doc:`/gallery/gravity_disturbance_topofree`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Topography-free (Bouguer) Gravity Disturbances</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="According to the Airy hypothesis of isostasy, topography above sea level is supported by a thickening of the crust (a root) while oceanic basins are supported by a thinning of the crust (an anti-root). Function harmonica.isostatic_moho_airy computes the depth to crust-mantle interface (the Moho) according to Airy isostasy. The function takes the depth to the crystalline basement and optionally any layers on top of it. Each layer is defined by its thickness and its density. In addition, one must assume a value for the reference thickness of the continental crust in order to convert the root/anti-root thickness into Moho depth. The function contains common default values for the reference thickness and crust, mantle [TurcotteSchubert2014]_.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_isostatic_moho_airy_thumb.png
    :alt:

  :doc:`/gallery/isostatic_moho_airy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Airy Isostatic Moho</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /gallery/gravity_disturbance
   /gallery/gravity_disturbance_topofree
   /gallery/isostatic_moho_airy

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

Forward Modeling
----------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The simplest geometry used to compute gravitational fields are point masses. Although modelling geologic structures with point masses can be challenging, they are very useful for other purposes: creating synthetic models, solving inverse problems, generating equivalent sources for interpolation, etc. The gravitational fields generated by point masses can be quickly computed either in Cartesian or in geocentric spherical coordinate systems. We will compute the gravitational acceleration generated by a set of point masses on a computation grid given in Cartesian coordinates using the harmonica.point_gravity function.">

.. only:: html

  .. image:: /gallery/forward/images/thumb/sphx_glr_point_gravity_thumb.png
    :alt:

  :doc:`/gallery/forward/point_gravity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Point Masses in Cartesian Coordinates</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One way to model three dimensional structures is to create a set of prisms that approximates their geometry and its physical properties (density, susceptibility, etc.). The harmonica.prism_layer offers a simple way to create a layer of prisms: a regular grid of prisms of equal size on the horizontal directions with variable top and bottom boundaries. It returns a xarray.Dataset with the coordinates of the centers of the prisms and their corresponding physical properties. The harmonica.DatasetAccessorPrismLayer Dataset accessor can be used to obtain some properties of the layer like its shape and size or the boundaries of any prism in the layer. The methods of this Dataset accessor can be used together with the harmonica.prism_gravity to compute the gravitational effect of the layer.">

.. only:: html

  .. image:: /gallery/forward/images/thumb/sphx_glr_prism_layer_thumb.png
    :alt:

  :doc:`/gallery/forward/prism_layer`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Layer of prisms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One possible application of the harmonica.prism_layer function is to create a model of the terrain and compute its gravity effect. Here we will use a regular grid of topographic and bathymetric heights for South Africa to create a prisms layer that model the terrain with a density of 2670 kg/m^3 and the ocean with a density contrast of -1900 kg/m^3 obtained as the difference between the density of water (1000 kg/m^3) and the normal density of upper crust (2900 kg/m^3). Then we will use harmonica.prism_gravity to compute the gravity effect of the model on a regular grid of observation points.">

.. only:: html

  .. image:: /gallery/forward/images/thumb/sphx_glr_prisms_topo_gravity_thumb.png
    :alt:

  :doc:`/gallery/forward/prisms_topo_gravity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gravitational effect of topography</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Computing the gravitational fields generated by regional or global scale structures require to take into account the curvature of the Earth. One common approach is to use spherical prisms also known as tesseroids. We will compute the downward component of the gravitational acceleration generated by a single tesseroid on a computation grid through the harmonica.tesseroid_gravity function.">

.. only:: html

  .. image:: /gallery/forward/images/thumb/sphx_glr_tesseroid_thumb.png
    :alt:

  :doc:`/gallery/forward/tesseroid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tesseroids (spherical prisms)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Layer of tesseroids">

.. only:: html

  .. image:: /gallery/forward/images/thumb/sphx_glr_tesseroid_layer_thumb.png
    :alt:

  :doc:`/gallery/forward/tesseroid_layer`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Layer of tesseroids</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The harmonica.tesseroid_gravity is capable of computing the gravitational effects of tesseroids whose density is defined through a continuous function of the radial coordinate. This is achieved by the application of the method introduced in [Soler2021]_.">

.. only:: html

  .. image:: /gallery/forward/images/thumb/sphx_glr_tesseroid_variable_density_thumb.png
    :alt:

  :doc:`/gallery/forward/tesseroid_variable_density`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tesseroids with variable density</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Grid transformations
--------------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reduction to the pole of a magnetic anomaly grid">

.. only:: html

  .. image:: /gallery/transformations/images/thumb/sphx_glr_reduction_to_pole_thumb.png
    :alt:

  :doc:`/gallery/transformations/reduction_to_pole`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reduction to the pole of a magnetic anomaly grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Total gradient amplitude of a regular grid">

.. only:: html

  .. image:: /gallery/transformations/images/thumb/sphx_glr_tga_thumb.png
    :alt:

  :doc:`/gallery/transformations/tga`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Total gradient amplitude of a regular grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tilt of a regular grid">

.. only:: html

  .. image:: /gallery/transformations/images/thumb/sphx_glr_tilt_thumb.png
    :alt:

  :doc:`/gallery/transformations/tilt`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tilt of a regular grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Upward continuation of a regular grid">

.. only:: html

  .. image:: /gallery/transformations/images/thumb/sphx_glr_upward_continuation_thumb.png
    :alt:

  :doc:`/gallery/transformations/upward_continuation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Upward continuation of a regular grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Upward derivative of a regular grid">

.. only:: html

  .. image:: /gallery/transformations/images/thumb/sphx_glr_upward_derivative_thumb.png
    :alt:

  :doc:`/gallery/transformations/upward_derivative`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Upward derivative of a regular grid</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

3D visualizations
-----------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The harmonica.prism_layer allows to easily create a layer of prisms whose top and bottom boundaries might drape a certain surface, like topography, bathymetry or the Moho discontinuity. It returns a xarray.Dataset object with the horizontal coordinates of the center of each prism, their top and bottom boundaries and their physical properties, like their density. Through the prism_layer accessor (see harmonica.DatasetAccessorPrismLayer) we can call some methods for our prism layer. For example, the harmonica.DatasetAccessorPrismLayer.gravity method lets us compute the gravitational fields of the layer on any set of observation points. Another interesting method is the harmonica.DatasetAccessorPrismLayer.to_pyvista: it converts the prism layer into a pyvista.UnstructuredGrid that could be easily plotted through pyvista.">

.. only:: html

  .. image:: /gallery/visualization/images/thumb/sphx_glr_prism_layer_pyvista_thumb.png
    :alt:

  :doc:`/gallery/visualization/prism_layer_pyvista`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plot a prism layer in 3D</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /gallery/equivalent_sources/index.rst
   /gallery/forward/index.rst
   /gallery/transformations/index.rst
   /gallery/visualization/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
