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

|h2| Variational Bayesian Inference: An Overview |h2e|

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
            x_i &\sim \mathcal{N}(\mu, \tau^{-1}) \\
            \tag{1}

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
           x_i &\sim \mathcal{N}(\mu_{z_i}, \sigma^2_{z_i})  \\
           \tag{2}

       In this case, you would want to (ideally) find an approximation to the
       joint distribution posterior (including both parameters and latent variables):
       :math:`q(\mu_1,\ldots,\mu_K, \sigma^2_1,\ldots, \sigma^2_K, \phi, z_1, \ldots, z_N)`,
       which most likely has some independence between these variables leading to
       several independent distributions.

       Note: I'm not sure if variational Bayes is actually used to solve the
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

* The Bayesian inference problem is hard and usually can't be solved analytically.
* Variational Bayes solves this problem by finding an approximate posterior
  distribution :math:`Q` that approximates the true posterior :math:`P`.
* It uses KL-divergence as a measure of how well our approximation fits the true posterior.
* The mean-field approximation partitions the unknown variables, and assumes each
  partition is independent and has a well-known, easy-to-analyze distribution.
* With some derivation, we can find an algorithm that iteratively computes
  the partitions of :math:`Q` by using the previous values of all the other
  partitions.

Now that we have an overview of this process, let's see how it actually works.

.. admonition:: Kullback-Leibeler Divergence

  `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__ 
  (aka information gain) is a non-symmetric measure of the difference between
  two probability distributions :math:`P` and :math:`Q`.  It is defined for discrete
  and continuous probability distributions as such:

  .. math::

    D_{KL}(P||Q) &= \sum_i P(i) \log \frac{P(i)}{Q(i)} \\
    D_{KL}(P||Q) &= \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx \\
    \tag{3}

  where :math:`p` and :math:`q` denote the densities of :math:`P` and :math:`Q`.

  There are several ways to intuitively understand KL-divergence, but let's use
  information entropy because I think it's a bit more intuitive.

  |h3| KL Divergence as Information Gain |h3e|

  To quickly summarize, entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`__)
  is the average amount of information or "surprise" for a probability
  distribution [1]_.  Entropy is defined as for both discrete and continuous distributions:

  .. math::

    H(P) &:= E_P[I_P(X)] = -\sum_{i=1}^n P(i) \log(P(i)) \\
    H(P) &:= E_P[I_P(X)] = - \int_{-\infty}^{\infty} p(x)\log(p(x)) dx \\
    \tag{4}

  An intuitive way to think about it is the (theoretical)
  minimum number of bits you need to encode an event (or symbol) drawn from your
  probability distribution (see `Shannon's source coding theorem
  <https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem>`_).
  For example, for a fair eight-sided die, each outcome is equi-probable, so we
  would need :math:`\sum_1^8 -\frac{1}{8}log_2(\frac{1}{8}) = 3` bits to encode
  the roll on average.  On the other hand, if we have a weighted eight-sided
  die where "8" came up 40 times more often than the other numbers, we would
  theoretically need about 1 bit to encode the roll on average (to get close,
  we would assign "8" to a single bit `0`, and others to something like `10`,
  `110`, `111` ... using a `prefix code <https://en.wikipedia.org/wiki/Prefix_code>`_).  
  
  In this way of viewing entropy, we're using the assumption that our symbols
  are drawn from probability distribution :math:`P` to get as close as we can
  to the theoretical minimum code length.  Of course, we rarely have an ideal encoding. 
  What would our average message length (i.e. entropy) be if we used the ideal
  symbols from another distribution such as :math:`Q`?  In that case, it would
  just be :math:`H(P,Q) := E_P[I_Q(X)] = E_P[-\log(Q(X))]`, which is also
  called the *cross entropy* of :math:`P` and :math:`Q`.  Of course, it would
  be larger than the ideal encoding, thus we would increase the average message
  length.  In other words, we need more information (or bits) to transmit this
  unoptimal :math:`Q` coding of the message.

  Thus, KL divergence can be viewed as this average extra-message length we need 
  when we wrongly assume the probability distribution, using :math:`Q` instead of
  :math:`P`:

  .. math::

    D_{KL}(P||Q) &= H(P,Q) - H(P)\\
                 &= -\sum_{i=1}^n P(i) \log(Q(i)) + \sum_{i=1}^n P(i) \log(P(i)) \\
                 &= \sum_{i=1}^n P(i) \log\frac{P(i)}{Q(i)} \\
                 \tag{5}

  You can probabily already see how this is a useful objective to try to minimize.  If
  we have some theoretic minimal distribution :math:`P`, we want to try to find an
  approximation :math:`Q` that tries to get as close as possible by minimizing
  the KL divergence.

  |h3| Forward and Reverse KL Divergence |h3e|

  One thing to note aboute KL divergence is that it's not symmetric, that is,
  :math:`D_{KL}(P||Q) \neq D_{K}(Q||P)`.  The former is called forward KL divergence,
  while the latter is called reverse KL divergence.  Let's start by looking at
  forward KL.  Taking a closer look at equation 5, we can see that when :math:`P`
  is large and :math:`Q \rightarrow 0`, the logarithm blows up.  This implies
  when choosing our approximate distribution :math:`Q` to minimize forward KL
  divergence, we want to "cover" all the non-zero parts of :math:`P` as best
  we can.  Figure 1 shows a good visualization of this.  

  .. figure:: /images/forward-KL.png
    :height: 300px
    :alt: Forward KL Divergence (source: `Eric Jang's Blog <http://blog.evjang.com/2016/08/variational-bayes.html>`__)
    :align: center

    Figure 1: Forward KL Divergence (source: `Eric Jang's Blog <http://blog.evjang.com/2016/08/variational-bayes.html>`__)

  From Figure 1, our original distribution :math:`P` is multimodal, while our
  approximate one :math:`Q` is bell shaped.  In the top diagram, if we just try
  to "cover" one of the humps, then the other hump of :math:`P` has a large
  mass with a near-zero value of :math:`Q`, resulting in a large KL divergence.
  In the bottom diagram, we can see that if we try to "cover" both humps by placing
  :math:`Q` somewhere in between, we'll get a smaller forward KL.  Of course,
  this has other problems like the maximum density (center of :math:`Q`) is now
  at a point that has low density in the original distribution.

  Now, let's take a look at reverse KL, where :math:`P` is still our theoretic
  distribution we're trying to match and :math:`Q` is our approximation:

  .. math::

    D_{KL}(Q||P) = \sum_{i=1}^n Q(i) \log\frac{Q(i)}{P(i)} \\
    \tag{6}

  From Equation 6, we can see that the opposite situation occurs.  If :math:`P`
  is small, we want :math:`Q` to be (proportioally) small too or the ratio
  might blow up.  Additionally, when :math:`P` is large, it doesn't cause us
  any particular problems because it just means the ratio is close to 0.
  Figure 2 shows this visually.
 
  .. figure:: /images/reverse-KL.png
    :height: 300px
    :alt: Reverse KL Divergence (source: `Eric Jang's Blog <http://blog.evjang.com/2016/08/variational-bayes.html>`__)
    :align: center

    Figure 2: Reverse KL Divergence (source: `Eric Jang's Blog <http://blog.evjang.com/2016/08/variational-bayes.html>`__)
  
  From Figure 2, we see in the top diagram that if we try to fit our unimodal
  distribution "in-between" the two maxima of :math:`P`, the tails cause us
  some problems where :math:`P` drops off much faster than :math:`Q` causing
  the ratio at those points to blow up.  The bottom diagram shows a better
  fit according to reverse KL, the tails of :math:`P` and :math:`Q` drop off
  at a similar rate, not causing any issues.  Additionally, since :math:`Q`
  matches one of the mode of our :math:`P` distribution well, the logarithm
  factor will be close to zero, also making for a better reverse KL fit.
  Reverse KL also has the nice property that the mode of our :math:`Q`
  distribution matches at least one of the modes of :math:`P`, which
  is really the best we could hope for with the shape of our approximation.

  In our use of KL divergence, we'll be using reverse KL divergence, not only
  because of the nice properties above, but for the more practical reason that
  the math works out nicely :p

|h2| From KL divergence to Optimization |h2e|

Remember what we're trying to accomplish: we have some intractable Bayesian
inference problem :math:`P(\theta|X)` we're trying to compute, where
:math:`\theta` are the unobserved variables (parameters or latent variables)
and :math:`X` are our observed data.  We could try to compute it directly using
Bayes theorem (continuous version, where :math:`p` is the density of
distribution :math:`P`):

.. math::

   p(\theta|X) &= \frac{p(X, \theta)}{p(X)} \\
               &= \frac{p(X|\theta)p(\theta)}{\int_{-\infty}^{\infty} p(X|\theta)p(\theta) d\theta} \\
               &= \frac{\text{likelihood}\cdot\text{prior}}{\text{marginal likelihood}} \\
               \tag{7}
        
However, this is generally difficult to compute because of the marginal
likelihood (sometimes called the evidence), but what if we didn't have to
directly compute the marginal likelihood and instead only needed the likelihood
(and prior)?  

This idea leads us to both the commonly used methods to solve
Bayesian inference problems: 
`MCMC <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__
and variational inference.  You can check out my previous post on MCMC but in general
it's quite slow since it involves repeated sampling but your approximation can
get arbitrarily close to the actual distribution (given enough time).
Variational inference on the other hand is a strict approximation that is much faster
because we can find an approximation via an optimizing problem.  It also can quantify
the lower bound on the marginal likelihood, which can help with model selection.

Now going back to our problem, we want to find an approximate distribution
:math:`Q` that minimizes (reverse) KL divergence.  Starting from reverse KL
divergence, let's do some manipulation to get to an equation
that's easy to interpret (using continuous version here), where our approximate
density is :math:`q(\theta)` and our theoretic one is :math:`p(\theta|X)`:

.. math::

  D_{KL}(Q||P) &= \int_{-\infty}^{\infty} q(\theta) \log\frac{q(\theta)}{p(\theta|X)} d\theta \\
               &= \int_{-\infty}^{\infty} q(\theta) \log\frac{q(\theta)}{p(\theta,X)} d\theta + 
                  \int_{-\infty}^{\infty} q(\theta) \log{p(X)} d\theta \\
               &= \int_{-\infty}^{\infty} q(\theta) \log\frac{q(\theta)}{p(\theta,X)} d\theta + 
                  \log{p(X)} \\
              \tag{8}

Where we're using Bayes theorem on line 2, rearranging we get:

.. math::

  \log{p(X)} &= D_{KL}(Q||P) 
            - \int_{-\infty}^{\infty} q(\theta) \log\frac{q(\theta)}{p(\theta,X)} d\theta \\
             &=  D_{KL}(Q||P) + \mathcal{L}(Q)
              \tag{9}

where :math:`\mathcal{L}` is called the (negative) *variational free energy* [2]_,
NOT the likelihood (I don't like the choice of symbols either but that's how it's shown
in most texts).  Recall that the evidence on the LHS is constant (for a given
model), thus if we maximize the variational free energy :math:`\mathcal{L}`, we
minimize (reverse) KL divergence as required.

This is the crux of variational inference: we don't need to explicitly compute
the posterior (or the marginal likelihood), we can solve an optimization
problem by finding the right distribution :math:`Q` that best fits our
variational free energy.  Note that we need to find a *function*, not just a
point, that maximizes :math:`\mathcal{L}`, which means we need to use
variational calculus (see my `past post <link://slug/the-calculus-of-variations>`__ 
on the subject), hence the name "variational Bayes".

|h2| The Mean-Field Approximation |h2e|

next section (try to prove the optimal constraint?)


|h2| Further Reading |h2e|

* Previous Posts: `Variational Calculus <link://slug/the-calculus-of-variations>`__, `Expectation-Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__, `Normal Approximation to the Posterior <link://slug/the-expectation-maximization-algorithm>`__, `Markov Chain Monte Carlo Methods, Rejection Sampling and the Metropolis-Hastings Algorithm <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__, `Maximum Entropy Distributions <link://slug/maximum-entropy-distributions>`__
* Wikipedia: `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__
* Machine Learning: A Probabilistic Perspective, Kevin P. Murphy
* `A Beginner's Guide to Variational Methods: Mean-Field Approximation <http://blog.evjang.com/2016/08/variational-bayes.html>`__, Eric Jung.

|br|

.. [1] There are a few different ways to intuitively understand information entropy.  See my previous post on `Maximum Entropy Distributions <link://slug/maximum-entropy-distributions>`__ for a slightly different explanation.

.. [2] The term variational free energy is from an alternative interpretation from physics.  As with a lot of ML techniques, this has its roots in physics where they make great use of probability to model the physical world. 
