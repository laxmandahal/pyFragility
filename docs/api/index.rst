API reference
=============

The top level (``import pyFragility as pf``) holds the everyday workflow. Other tools live in
submodules, reached as ``pf.inference.bootstrap``, ``pf.risk.HazardCurve`` and so on.

.. toctree::
   :maxdepth: 1
   :hidden:

   pyfragility
   inference
   bayes
   risk
   binomial
   capacity
   cloud
   ordinal
   engine
   links
   datasets
   plotting
   fragility
   mle
   glm
   variance
   likelihood

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Page
     - Contents
   * - :doc:`pyfragility`
     - ``fit_msa``, ``fit_field_data``, ``fit_ida``, ``fit_cloud``, ``fit_damage_states``,
       ``FragilityFit``, ``HazardCurve``, risk and reporting functions
   * - :doc:`inference`
     - Misspecification test, goodness of fit, bootstrap, profile likelihood
   * - :doc:`bayes`
     - Bayesian sampling and priors
   * - :doc:`risk`
     - Hazard curves, mean annual frequency, expected loss (paper-era functions included)
   * - :doc:`binomial`, :doc:`capacity`, :doc:`cloud`, :doc:`ordinal`
     - The likelihood classes behind each data type
   * - :doc:`engine`
     - ``Likelihood`` (the extension point), fitting and covariance machinery
   * - :doc:`links`
     - Probit, logit and complementary log-log links
   * - :doc:`datasets`
     - The paper's example data
   * - :doc:`plotting`
     - Plot functions
   * - :doc:`fragility`, :doc:`mle`, :doc:`glm`, :doc:`variance`, :doc:`likelihood`
     - Reference implementation of the paper, against which the test suite compares the
       general code
