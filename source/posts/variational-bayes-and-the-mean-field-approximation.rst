.. title: Variational Bayes and The Mean-Field Approximation
.. slug: variational-bayes-and-the-mean-field-approximation
.. date: 2017-03-02 08:02:46 UTC-05:00
.. tags: Bayesian, variational calculus, mean-field, Kullback-Leibeler, mathjax
.. category: 
.. link: 
.. description: A brief introduction to variational Bayes and the mean-field approximation.
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

This post is going to cover Variational Bayesian methods and, in particular,
the most common one, the mean-field approximation.  This is a topic that I've
been trying to get to for a while now but didn't quite have all the background
that I needed.  After picking up the main ideas from
`Variational Calculus <link://slug/the-calculus-of-variations>`__ and
getting more fluent in manipulating probability statements like
in my `EM <link://slug/the-expectation-maximization-algorithm>`__ post,
this variational Bayes stuff seems a lot easier.

Variational Bayesian methods are a set of techniques to approximate posterior
distributions in `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__.
If this sounds a bit terse, keep reading!  I hope to provide a bunch of intuition
so that the big ideas are easy to understand (which they are), but of course we 
can't do that well unless we have a healthy dose of mathematics.  For some of the
background concepts, I'll try to refer you to good sources (including my own),
which I find is the main blocker to understanding this subject.  Enjoy!

.. TEASER_END

|h2| Variational Bayes: An Overview |h2e|

Before we get into the nitty-gritty of it all, let's just go over at a high level
what we're trying to do.  First, we're trying to perform `Bayesian inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__, which basically means given a model, we're trying
to find distributions for the unobserved variables (either parameters or latent
variables since they're treated the same).  This problem usually involves
hard-to-solve integrals with no analytical solution.  

There are two main avenues to solve this problem.  The first is to just get a
point-estimate for each of the unobserved variables (either MAP or mean) but
this is not ideal since we can't quantify the uncertainty of the unknown
variables (and is kind of against the spirit of Bayesian analysis).  The other
aims to find a distribution of each unknown variable.  One good but relatively
slow method is to use `MCMC <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__ to iteratively draw samples that eventually give you the shape
of every unknown variable.  Another is to use variational Bayes helps to find
an approximation of the distribution of the unknown variables.  With variational
Bayes, you only get an approximation but it's in the form of a distribution!
So long as your approximation is pretty good, you can do all the nice Bayesian
analysis you like.  

The next example shows a couple of Bayesian inference problems to make things
more concrete.

.. admonition:: Example 1: Bayesian Inference Problems

    1. **Fitting a Gaussian with unknown mean and variance**: 
       Given observed data :math:`X=\{x_1,\ldots, x_N\}`, we wish to model this data
       as a normal distribution with parameters :math:`\mu,\sigma^2` with priors
       a normally distributed prior on the mean and an inverse-gamma
       distributed prior on the variance.  More precisely, our model can be defined as:

       .. math::

            \mu &\sim \mathcal{N}(\mu_0, (\lambda_0\tau)^{-1}) \\ 
            \tau &\sim \text{Gamma}(a_0, b_0) \\
            x_i &\sim \mathcal{N}(\mu, \tau^{-1})

       where the hyperparameters :math:`\mu_0, \lambda_0, a_0, b_0` are given.
       In this model, the variables :math:`\mu,\tau` are unobserved, so
       we would use variational Bayes to approximate the posterior
       distribution :math:`q(\mu, \tau) =p(\mu, \tau | x_1, \ldots, x_N)`
       for the parameters :math:`\mu` and :math:`\tau`.

    2. **Bayesian Gaussian Mixture Model**:
       Given a 
       `Bayesian Gaussian mixture model <https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model>`__ 
       with :math:`K` mixture components
       and :math:`N` observations :math:`\{x_1,\ldots,x_N\}`, latent variables
       :math:`\{z_1,\ldots,z_N\}`, parameters :math:`\{\mu_i, \sigma^2_i, \phi, \nu\}`
       and hyperparameters :math:`\{\mu_0, \lambda, \sigma^2_0, \beta\}`:

       .. math::

           \mu_{i=1,\ldots,K} &\sim \mathcal{N(\mu_0, \lambda\sigma_i^2)} \\
           \sigma^2_{i=1,\ldots,K} &\sim \text{Inverse-Gamma}(\nu, \sigma_0^2) \\
           \phi &\sim \text{Symmetric-Dirichlet}_K(\beta) \\
           z_i &\sim \text{Categorical}(\phi) \\
           x_i &\sim \mathcal{N}(\mu_{z_i}, \sigma^2_{z_i})

       In this case, you would want to (ideally) find an approximation to the
       joint distribution posterior (including both parameters and latent variables):
       :math:`q(\mu_1,\ldots,\mu_K, \sigma^2_1,\ldots, \sigma^2_K, \phi, z_1, \ldots, z_N)`,
       which most likely has some independence between these variables leading to
       several independent distributions.

       Note: I'm not sure it variational Bayes is actually used to solve the
       Gaussian mixture model in practice (although theoretically it can be).  It might be
       more appropriate to use `MCMC <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__.


Now that we know our problem, next thing we need to is define what it means to
be a good approximation.  In many of these cases,
`Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__ (KL divergence)
is a non-symmetric measure of the difference between two probability distributions
:math:`P` and :math:`Q`.  We'll discuss this in a bit more detail below but the way
we set up the problem will be with :math:`P` being the true posterior distribution,
and :math:`Q` being the approximate distribution.  With a bit of math, we can
get to an iterative algorithm to find :math:`Q`.

Next, we assume our approximate distribution :math:`Q` takes the form of some
well-known and easy-to-analyze distributions.  In the mean-field approximation (a common
type of variational Bayes),
we assume that the unknown variables can be partitioned so that each partition
is independent of the others (a simplifying assumption).  Using KL divergence,
we can derive mutually dependent equations (one for each partition) that define
the shape of :math:`Q`.  The leads to an easy-to-compute iterative algorithm
(similar to the `EM algorithm <link://slug/the-expectation-maximization-algorithm>`__)
where we use all other previously calculated partitions to derive the current one.

To summarize, variational Bayes has these ideas:

* Bayesian inference problem is hard and usually can't be solved analytically.
* Variational Bayes solves this problem by finding an approximate posterior
  distribution :math:`Q` that approximates the true posterior :math:`P`.
* It uses KL-divergence as a measure of how well our approximation fits the true posterior.
* The mean-field approximation partitions the unknown variables and assumes each
  partition has a well-known easy-to-analyze distribution.
* With some derivation, we can find an algorithm that iteratively computes
  the partitions of :math:`Q` by using the previous values of all the other
  partitions.

Now that we have an overview of this process, let's see how it actually works.

|h2| Kullback-Leibeler Divergence and Finding Like Probability Distributions |h2e|



|h2| Further Reading |h2e|

* Previous Posts: `Variational Calculus <link://slug/the-calculus-of-variations>`__, `Expectation-Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__, `Normal Approximation to the Posterior <link://slug/the-expectation-maximization-algorithm>`__,
`Markov Chain Monte Carlo Methods, Rejection Sampling and the Metropolis-Hastings Algorithm <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__

* Wikipedia: `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__
