.. title: Bayesian Learning via Stochastic Gradient Langevin Dynamics and Bayes by Backprop
.. slug: bayesian-learning-via-stochastic-gradient-langevin-dynamics-and-bayes-by-backprop
.. date: 2022-11-23 21:25:40 UTC-05:00
.. tags: Bayesian, Bayes by Backprop, SGLD, variational inference, elbo, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

After a long digression, I'm finally back to one of the main lines of research
that I wanted to write about.  The two main ideas in this post are not that
recent but have been quite impactful (one of the 
`papers <https://icml.cc/virtual/2021/test-of-time/11808>`__ won a recent ICML
test of time award).  They address two of the topics that are near and dear to
my heart: Bayesian learning and scalability.  I dare ask wouldn't be interested
in the intersection of such topics?  In any case, I hope you enjoy my
explanation of it.

This post is about two techniques to perform scalable Bayesian inference.  They
both address the problem using stochastic gradient descent (SGD) but in very
different ways.  One leverages the observation that SGD plus some noise will
converge to Bayesian posterior sampling [Welling2011]_, while the other generalizes the
"reparameterization trick" from variational autoencoders to enable non-Gaussian
posterior approximations [Blundell2015]_.  Both are easily implemented in the modern deep
learning toolkit thus benefit from the massive scalability of that toolchain.
As usual, I go over the necessary background (or refer you to my previous
posts), intuition, some math, and a couple of toy examples that I implemented.



.. TEASER_END
.. section-numbering::
.. raw:: html

    <div class="card card-body bg-light">
    <h1>Table of Contents</h1>

.. contents:: 
    :depth: 2
    :local:

.. raw:: html

    </div>
    <p>


Motivation
==========

Bayesian learning is all about learning the `posterior <https://en.wikipedia.org/wiki/Posterior_probability>`__ 
distribution of the statistical parameters of your model, which in turns allows
you to quantify the uncertainty about them.  The classic place to start is with
Bayes theorem:

.. math::

   p({\bf \theta}|{\bf x}) &= \frac{p({\bf x}|{\bf \theta})p({\bf \theta})}{p({\bf x})} \\
                           &= \text{const}\cdot p({\bf x}|{\bf \theta})p({\bf \theta}) \\
                           &= \text{const}\cdot \text{likelihood} \cdot \text{prior} \\
                           \tag{1}

where :math:`{\bf x}` is a vector of data points (often 
`IID <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`__)
and :math:`{\bf \theta}` is the vector of statistical parameters of your model.

Generally, there is no closed form and you have to resort to heavy methods such
as Markov Chain Monte Carlo (MCMC) methods or some form of approximation (which
we'll get to later).  MCMC never give you an exact closed form but instead give
you either samples from the posterior distribution, which you can use then use
to compute any statistics you like.  These methods are quite slow because they
rely on `Monte Carlo <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__
methods, which require repeated random sampling. 

This brings us to our first scalability problem Bayesian learning: it does not
scale well with the number of parameters.  Randomly sampling with MCMC implies
that you have to "walk" the parameter space, which potentially grows
exponentially with the number of parameters.  There are many techniques to make
this more efficient but ultimately it's hard to compensate for an exponential.
The natural model for this situation is neural networks which can have orders
of magnitude more parameters compared to classic Bayesian learning problems
(I'll also add that the use-case of the posterior is usually different too).

The other non-obvious scalability issue with MCMC from Equation 1 is the data.
Each evaluation of MCMC requires an evaluation of the likelihood and prior from
Equation 1.  For large data (e.g. modern deep learning datasets), you quickly
hit issues either with memory and/or computation speed.

Modern deep learning has really solved both of these problems by leveraging the
one of the simplest optimization method out there (stochastic gradient descent)
along with the massive compute power of modern hardware (and its associated
toolchain).  How can we leverage these developments to scale Bayesian learning?
Keep reading to found out!

Background
==========

Bayesian Hierarchical Models and Bayesian Networks
--------------------------------------------------

We can take the idea of parameters and prior from Equation 1 to multiple
levels.  Equation 1 implicitly assumes that there is one "level" of parameters
(:math:`\theta`) that we're trying to estimate with prior distributions
(:math:`p({\bf \theta})`) attached to them, but there's no reason why you only
need a single level.  In fact, our parameters can be conditioned on parameters,
which can be conditioned on parameters, and so on.  
This is called `Bayesian hierarchical modeling <https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling>`__.
If this sounds oddly familiar, it's the same thing as `Bayesian networks
<https://en.wikipedia.org/wiki/Bayesian_network#Graphical_model>`__ in a different context (if you're
familiar with that).  My `previous post <link://slug/the-expectation-maximization-algorithm>` that gives a nice high
level summary on the intuition with latent variables.

To quickly summarize, in a parameterized statistical model there are broadly
two types of variables: observed and unobserved.  Observed are the ones
where we have values for often with multiple observations where we assume
they are `IID <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`__.

Unobserved variables can have different names. In Bayesian networks they
are usually called latent or hidden (random) variables, which can have 
complex conditional dependencies specified as a DAG.  In hierarchical models
they are called **hyperparameters**, which are the parameters of the 
observed models, the parameters of parameters, parameters of parameters of
parameters and so on.  Similarly, each of these hyperparameters has a 
distribution which we call a **hyperprior**.  

These two concepts are mathematically the same and from what I gather really
on vary based on the context.  In the context of hierarchical models,
the hyperparameters and hyperpriors represent some structural knowledge
about the problem, hence of the use of term "priors".  This view is more
typical in terms of Bayesian statistics where the number of stages (and thus
variables) is usually small (two or three).

In Bayesian networks, the latent variables can represent the underlying
phenomenon but also can be artificially introduced to make the problem more
tractable.  This happens more often in machine learning e.g. `variational
autoencoders <link://slug/variational-autoencoders>`__.  In these contexts,
they are often modeling a much bigger network and can have arbitrarily larger
stages and network size.  With varying assumptions on the latent variables and
their connectivity, there are many efficient algorithms that can perform either
approximate or exact inference on them.

.. admonition:: Example 1: Hierarchical Model

   ()


Markov Chain Monte Carlo and Langevin Dynamics
----------------------------------------------
- Metropolis Hastings
- Langevin Dynamics
- HMC?
- LMC

Stochastic Gradient Descent and RMSProp
---------------------------------------

- SGD
- SGD guarantees
- RMSProp 

Variational Inference
---------------------

- VI, q-approx function
- ELBO
- Reparameterization trick

Stochastic Gradient Langevin Dynamics 
=====================================

- Explain intuition
- Proof of correctness

Bayes by Backprop
=================

- Used in neural networks
- Still uses VI

Experiments
===========

Simple Gaussian Mixture
-----------------------

Stochastic Volatility Model
---------------------------

Conclusion
==========

References
==========
* Wikipedia: test 1

.. [Welling2011] Max Welling and Yee Whye Teh, "`Bayesian Learning via Stochastic Gradient Langevin Dynamics <https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf>`__", ICML 2011.
.. [Blundell2015] Blundell et. al, "`Weight Uncertainty in Neural Networks <https://arxiv.org/abs/1505.05424>`__", ICML 2015.
.. [Li] Li et. al, "`Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks <https://arxiv.org/abs/1512.07666>`__", AAAI 2016.
.. [Ma] Yi-An Ma, Tianqi Chen, Emily B. Fox, "`A Complete Recipe for Stochastic Gradient MCMC <https://arxiv.org/abs/1506.04696>`__", NIPS 2015.
