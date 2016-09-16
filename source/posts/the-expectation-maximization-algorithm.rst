.. title: The Expectation-Maximization Algorithm
.. slug: the-expectation-maximization-algorithm
.. date: 2016-09-12 07:47:47 UTC-04:00
.. tags: expectation-maximization, latent models, gaussian mixture models, mathjax
.. category: 
.. link: 
.. description: An overview of the expectation-maximization algorithm
.. type: text


.. |br| raw:: html

   <br />

.. |H2| raw:: html

   <br/><h3>

.. |H2e| raw:: html

   </h3>

.. |H3| raw:: html

   <h4>

.. |H3e| raw:: html

   </h4>

.. |center| raw:: html

   <center>

.. |centere| raw:: html

   </center>

This post is going to talk about a widely used method to find either the
maximum likelihood (MLE) or maximum a posteriori (MAP) estimate of parameters
in latent variable models.  It's a very widely used algorithm with perhaps the 
most famous variant being the one used in the k-means algorithm.
Even though it's so ubiquitous, whenever I've tried to understand *why* this
algorithm works, using a variety of sources from the web such as
lecture notes or papers, I never quite got the intuition right.  Now a bit
wiser, I'm going to *attempt* to explain the algorithm hopefully with a bit
more clarity by going back to the basics starting with latent variable models
and the likelihood function, then moving on showing the math relating to a
simple Gaussian mixture model [1]_. 

.. TEASER_END

|h2| Background |h2e|

|h3| Latent Variables |h3e|

A latent variable model is a type of statistical model that contains two types
of variables: *observed variables* and *latent variables*.  Observed variables
are ones that we can measure or record, while latent (sometimes called
*hidden*) variables are ones that we cannot directly observe but rather
inferred from the observed variables. 

One reason why we add latent variables is to model "higher level concepts"
(i.e. latent variables) in the data, usually these "concepts" are unobserved
but easily understood by the modeller.  Adding these variables can simplify
our model by reducing the number of parameters we have to estimate.  Consider
the problem of modelling medical symptoms such as blood pressure, heart rate
and glucose levels (observed outcomes) and mediating factors such as smoking,
diet and exercise (observed "inputs").  We could model all the possible
relationships between the mediating factors and observed outcomes but the
number of connections grows very quickly.
Instead, we can model this problem as having the mediating factors
causing a non-observable hidden variable such as heart disease, which
in turn causes our medical symptoms.  This is shown in the next figure.

.. image:: /images/latent_vars.png
   :height: 300px
   :alt: Latent Variables
   :align: center

Notice that the number of connections now grows linearly instead of
multiplicatively as you add more factors, this greatly reduces the number of
parameters you need to estimate.  In general, you can have an arbitrary number
of connections between variables with as many latent variables as you wish.
These models are more generally known as `Probabilistic graphical models (PGMs)
<https://en.wikipedia.org/wiki/Graphical_model>`_.  We'll be focusing on a much
simpler case however, as explained in the next section.

|h3| Gaussian Mixture Models |h3e|

As an example, suppose we're trying to understand the prices of houses across
the city.  The housing price will be heavily dependent on the neighbourhood,
that is, houses clustered around a neighbourhood will be close to the average
price of the neighbourhood.
In this context, it is straight forward to observe the prices at which houses
are sold (observed variables) but what is not so clear is how is to observe or
estimate the price of a "neighbourhood" (the latent variables).  A simple model
for modelling the neighbourhood price is using a Gaussian (or normal)
distribution, but which house prices should be used to estimate the average
neighbourhood price?  Should all house prices be used in equal proportion, even
those on the edge?  What if a house is on the border between two
neighbourhoods?  These are all great questions that lead us to a particular
type of latent variable model called a Gaussian mixture model.

Following along with this housing price example, let's represent the price of
each house as random variable :math:`x_i` and the unobserved neighbourhood 
it belongs to as :math:`z_i` [2]_.  Further, let's suppose we have :math:`K`
neighbourhoods, each of which we model using a Gaussian :math:`\mathcal{N}(\mu_k,
\sigma_k^2)` with mean :math:`\mu_k` and variance :math:`\sigma_k^2`.   Then,
the density of :math:`x_i` is given by:

.. math::

    p(x_i|\theta) &=  \sum_{z=1}^K p(z_i=z) p(x_i| z_i=z, \mu_k, \sigma_k^2)  \\
    x_i| z_i &\sim \mathcal{N}(\mu_k, \sigma_k^2) \\
    z_i &\sim \text{Categorical}(\pi_i) \tag{1}
    
Where :math:`\theta` are the parameters of the Gaussians (all the :math:`\mu_k,
\sigma_k^2`).  

Notice that since :math:`z_i` variables are non-observed, we need
to `marginalize <https://en.wikipedia.org/wiki/Marginal_distribution>`_ them out
to get the density of the observed variables.
Translating Equation 1 to plainer language: we model the price distribution of each
house as a linear combination [3]_ of our :math:`K` Gaussians (neighbourhoods).

Now we have a couple of inference problems:

* Computing responsibility of each cluster to a point :math:`x_i`; and
* Estimating the parameters of the Gaussians

TODO EXPLAIN THIS MORE

Rephrasing the questions we posed above: we now have a bunch of observations
:math:`x_i` (housing prices), and we want to estimate our :math:`K` Gaussian
distributions :math:`\mathcal{N}(\mu_k, \sigma_k^2)` (neighbourhoods) i.e.
estimate all the parameters of :math:`\theta` (:math:`\mu_k` and
:math:`\sigma_k^2`).  


Turns out that there is a relatively simple algorithm for
finding the MLE (or MAP) estimate for this problem called the
Expectation-Maximization algorithm [4]_.


|h2| The Expectation-Maximization Algorithm |h2e|

* Need hidden variables to discrete

|h2| Further Reading |h2e|

* Wikipedia: 
  `Expectation-Maximization algorithm <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_,
  `Mixture Models <https://en.wikipedia.org/wiki/Mixture_model>`_
* Machine Learning: A Probabilistic Perspective, Kevin P. Murphy

|br|

.. [1] The material in this post is heavily based upon the treatment in *Machine Learning: A Probabilistic Perspective* by Kevin P.  Murphy; it has a much more detailed explanation and I encourage you to check it out.

.. [2] This is actually only one application of Gaussian Mixture Models.  Another common one is using it as a generative classifier i.e. estimating :math:`p(X_i, y_i)` (where we label :math:`z_i` as :math:`y_i` as per convention for classifiers).  Since both :math:`X_i` and :math:`y_i` are observable, it's much easier to directly estimate the density versus the case where we have to infer values for hidden variables.

.. [3] I say linear combination because we don't actually know the value of :math:`z_i`, so one way to think about it is the expected value of :math:`z_i`.  This translates to :math:`x_i` having a portion of each of the :math:`K` Gaussians being responsible for generating it.  Thus, the linear combination idea.

.. [4] To perform full Bayesian analysis and get the full posterior distribution, you would probably require something more complicated like MCMC, which I've explained in a `previous post <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`_.
