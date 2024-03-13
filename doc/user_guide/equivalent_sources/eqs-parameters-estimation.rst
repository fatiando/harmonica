.. _eqs-parameters-estimation:

Estimating damping and depth parameters
=======================================

When interpolating gravity and magnetic data through the
:ref:`equivalent_sources` technique we need to choose values for some
parameters, like the ``depth`` at which the sources will be located or the
amount of ``damping`` that should be applied.

The choice of these *hyperparameters* can significantly affect the accuracy of
the predictions. One way to make this choice could be a visual inspection of
the predictions, but that could be tedious and non-objective. Instead, we could
estimate these hyperparameters by evaluating the performance of the equivalent
sources with different values for each hyperparameter through a **cross
validation**.

.. seealso::

   Evaluating the performance of :class:`EquivalentSources` through cross
   validation is very similar to how we do it for any :mod:`verde` gridder.
   Refer to Verde's
   `Evaluating Performance
   <https://www.fatiando.org/verde/latest/tutorials/model_evaluation.html>`_
   and `Model Selection
   <https://www.fatiando.org/verde/latest/tutorials/model_selection.html>`_
   for further details.


Cross-validating equivalent sources
-----------------------------------

Lets start by loading some sample gravity data:

.. jupyter-execute::

   import ensaio
   import pandas as pd

   fname = ensaio.fetch_bushveld_gravity(version=1)
   data = pd.read_csv(fname)
   data

And project their coordinates using a Mercator projection:

.. jupyter-execute::

   import pyproj

   projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.values.mean())
   easting, northing = projection(data.longitude.values, data.latitude.values)

   coordinates = (easting, northing, data.height_geometric_m.values)

Lets fit the gravity disturbance using equivalent sources and a first guess for
the ``depth`` and ``damping`` parameters.

.. jupyter-execute::

   import harmonica as hm

   eqs_first_guess = hm.EquivalentSources(depth=1e3, damping=1)
   eqs_first_guess.fit(coordinates, data.gravity_disturbance_mgal)


We can use a **cross-validation** to evaluate how well these equivalent sources
can accurately predict the values of the field on unobserved locations.
We will use :func:`verde.cross_val_score` and then we will compute the mean
value of the score obtained after each cross validation.

.. jupyter-execute::

   import numpy as np
   import verde as vd

   score_first_guess = np.mean(
       vd.cross_val_score(
           eqs_first_guess,
           coordinates,
           data.gravity_disturbance_mgal,
       )
   )
   score_first_guess

The resulting score corresponds to the R^2. It represents how well the
equivalent sources can reproduce the variation of our data. As closer it gets
to one, the better the quality of the predictions.


Estimating hyperparameters
--------------------------

We saw that we can evaluate the performance of some equivalent sources with
some values for the ``depth`` and ``damping`` parameters through cross
validation.
Now, lets use it to estimate a set of hyperparameters that produce more
accurate predictions.
To do so we are going to apply a simple grid search over the ``depth``,
``damping`` space, apply cross validation for each pair of values and keeping
track of their score.


Lets start by defining some possible values of ``damping`` and ``depth`` to
explore:

.. jupyter-execute::


   dampings = [0.01, 0.1, 1, 10,]
   depths = [5e3, 10e3, 20e3, 50e3]

.. note::

   The actual value of the damping is not significant as its order of
   magnitude. Exploring different powers of ten is a good place to start.

Then we can build a ``parameter_sets`` list where each element corresponds to
each possible combination of the values of ``dampings`` and ``depths``:

.. jupyter-execute::

   import itertools

   parameter_sets = [
       dict(damping=combo[0], depth=combo[1])
       for combo in itertools.product(dampings, depths)
   ]
   print("Number of combinations:", len(parameter_sets))
   print("Combinations:", parameter_sets)

And now we can actually ran one cross validation for each pair of parameters:

.. jupyter-execute::

   equivalent_sources = hm.EquivalentSources()

   scores = []
   for params in parameter_sets:
       equivalent_sources.set_params(**params)
       score = np.mean(
           vd.cross_val_score(
               equivalent_sources,
               coordinates,
               data.gravity_disturbance_mgal,
           )
       )
       scores.append(score)
   scores

Once every score has been computed, we can obtain the best score and the
corresponding parameters that generate it:

.. jupyter-execute::

   best = np.argmax(scores)
   print("Best score:", scores[best])
   print("Score with defaults:", score_first_guess)
   print("Best parameters:", parameter_sets[best])

We have actually improved our score!

Finally, lets grid the gravity disturbance data using the equivalent sources of
the first guess and the best ones obtained after cross validation.


Create some equivalent sources out of the best set of parameters:

.. jupyter-execute::

   eqs_best = hm.EquivalentSources(**parameter_sets[best]).fit(
       coordinates, data.gravity_disturbance_mgal
   )

And grid the data using the two equivalent sources:

.. jupyter-execute::

   # Define grid coordinates
   region = vd.get_region(coordinates)
   grid_coords = vd.grid_coordinates(
       region=region,
       spacing=2e3,
       extra_coords=2.5e3,
   )

   grid_first_guess = eqs_first_guess.grid(grid_coords)
   grid = eqs_best.grid(grid_coords)

Lets plot it:

.. jupyter-execute::

   import pygmt

   # Set figure properties
   w, e, s, n = region
   fig_height = 10
   fig_width = fig_height * (e - w) / (n - s)
   fig_ratio = (n - s) / (fig_height / 100)
   fig_proj = f"x1:{fig_ratio}"

   maxabs = vd.maxabs(grid_first_guess.scalars, grid.scalars)

   fig = pygmt.Figure()

   # Make colormap of data
   pygmt.makecpt(cmap="polar+h0",series=(-maxabs, maxabs,))

   title = "Gravity disturbance with first guess"

   fig.grdimage(
      projection=fig_proj,
      region=region,
      frame=[f"WSne+t{title}", "xa100000+a15", "ya100000"],
      grid=grid_first_guess.scalars,
      cmap=True,
   )
   fig.colorbar(cmap=True, frame=["a50f25", "x+lmGal"])

   fig.shift_origin(xshift=fig_width + 1)

   title = "Gravity disturbance with best params"

   fig.grdimage(
      frame=[f"ESnw+t{title}", "xa100000+a15", "ya100000"],
      grid=grid.scalars,
      cmap=True,
   )
   fig.colorbar(cmap=True, frame=["a50f25", "x+lmGal"])

   fig.show()

The best parameters not only produce a better score, but they also generate
a visibly more accurate prediction. In the first plot the equivalent sources
are so shallow that we can actually see the distribution of sources in the
produced grid.

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <eqs-parameters-estimation>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <eqs-parameters-estimation>`
        :text-align: center
