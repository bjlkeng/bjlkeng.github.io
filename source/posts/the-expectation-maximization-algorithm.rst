.. title: The Expectation-Maximization Algorithm
.. slug: the-expectation-maximization-algorithm
.. date: 2016-09-12 07:47:47 UTC-04:00
.. tags: expectation-maximization, latent models, gaussian mixture models
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
simple Gaussian mixture model.  The material in this post is heavily based upon
the treatment in *Machine Learning: A Probabilistic Perspective* by Kevin P.
Murphy; it has a much more detailed explanation and I encourage you to check it out.

.. TEASER_END

|h2| Latent Variables and Gaussian Mixture Models |h2e|

A latent variable model is a type of statistical model that contains two types
of variables: *observed variables* and *latent variables*.  Observed variables
are ones that we can measure or record, while latent (sometimes called
*hidden*) variables are ones that we cannot directly observe but rather
inferred from the observed variables. 

One reason why we add latent variables is to model "higher level concepts"
(i.e. latent variables) in the data, usually these "concepts" are unobserved
but easily understood by the modeller.  In certain application, learning these
concepts can regarded as a type of unsupervised learning.  Another reason we
add thse variables is that is reduces the number of parameters we have to
estimate, simplifying the model.  The next image shows a complex distribution
that can easily be modelled using three Gaussians
(image taken from http://dirichletprocess.weebly.com/clustering.html).

.. image:: /images/gmm.png
   :height: 250px
   :alt: Mixtures of Gaussian
   :align: center

As an example, suppose we're trying to understand the prices of houses across
the city.  The housing price will be heavily dependent on the neighbourhood,
that is, houses clustered around a neighbourhood will be close to the average
price of the neighbourhood.
In this context, it is straight forward to observe the prices at which houses
are sold but what is not so clear is how is to estimate the price of a
"neighbourhood".  A natural model for modelling the neighbourhood is using a
Gaussian (or normal) distribution, but which house prices should be used to
estimate the average neighbourhood price?  Should all house prices be used in
equal proportion, even those on the edge?  What if a house is on the border
between two neighbourhoods?  These are all great questions that lead us
to a particular type of latent variable model called a Gaussian mixture model.





|h2| Further Reading |h2e|

* Wikipedia: 
  `Expectation-Maximization algorithm <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_,
  `Mixture Models <https://en.wikipedia.org/wiki/Mixture_model>`_
* Machine Learning: A Probabilistic Perspective, Kevin P. Murphy

|br|

.. [1] Footnote 1
