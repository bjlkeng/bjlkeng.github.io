.. title: Variational Bayes and The Mean-Field Approximation
.. slug: variational-bayes-and-the-mean-field-approximation
.. date: 2017-03-04 08:02:46 UTC-05:00
.. tags: Bayesian, variational calculus, mean-field, Kullback-Leibler, mathjax
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
been trying to understand for a while now but didn't quite have all the background
that I needed.  After picking up the main ideas from
`variational calculus <link://slug/the-calculus-of-variations>`__ and
getting more fluent in manipulating probability statements like
in my `EM <link://slug/the-expectation-maximization-algorithm>`__ post,
this variational Bayes stuff seems a lot easier.

Variational Bayesian methods are a set of techniques to approximate posterior
distributions in `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__.
If this sounds a bit terse, keep reading!  I hope to provide some intuition
so that the big ideas are easy to understand (which they are), but of course we 
can't do that well unless we have a healthy dose of mathematics.  For some of the
background concepts, I'll try to refer you to good sources (including my own),
which I find is the main blocker to understanding this subject (admittedly, the
math can sometimes be a bit cryptic too).  Enjoy!

.. TEASER_END

|h2| Variational Bayesian Inference: An Overview |h2e|

Before we get into the nitty-gritty of it all, let's just go over at a high level
what we're trying to do.  First, we're trying to perform `Bayesian inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__, which basically means given a model, we're trying
to find distributions for the unobserved variables (either parameters or latent
variables since they're treated the same).  This problem usually involves
hard-to-solve integrals with no analytical solution.  

There are two main avenues to solve this problem.  The first is to just get a
point-estimate for each of the unobserved variables (either MAP or MLE) but
this is not ideal since we can't quantify the uncertainty of the unknown
variables (and is against the spirit of Bayesian analysis).  The other
aims to find a (joint) distribution of each unknown variable.  With a proper
distribution for each variable, we can do a whole bunch of nice Bayesian
analysis like the mean, variance, 95% credible interval etc.

One good but relatively slow method for finding a distribution is to use `MCMC
<link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__
(a simulation technique)
to iteratively draw samples that eventually give you the shape of the joint distribution 
of the unknown variables.  Another method is to use variational Bayes, which helps to
find an approximation of the distribution in question.  With variational Bayes,
you only get approximation but it's in analytical form (read: easy to compute).  So
long as your approximation is pretty good, you can do all the nice Bayesian
analysis you like, and the best part is it's relatively easy to compute!

The next example shows a couple of Bayesian inference problems to make things
more concrete.

.. admonition:: Example 1: Bayesian Inference Problems

    1. **Fitting a univariate Gaussian with unknown mean and variance**: 
       Given observed data :math:`X=\{x_1,\ldots, x_N\}`, we wish to model this data
       as a normal distribution with parameters :math:`\mu,\sigma^2` with a
       normally distributed prior on the mean and an inverse-gamma distributed
       prior on the variance.  More precisely, our model can be defined as:

       .. math::

            \mu &\sim \mathcal{N}(\mu_0, (\kappa_0\tau)^{-1}) \\ 
            \tau &\sim \text{Gamma}(a_0, b_0) \\
            x_i &\sim \mathcal{N}(\mu, \tau^{-1}) \\
            \tag{1}

       where the hyperparameters :math:`\mu_0, \kappa_0, a_0, b_0` are given
       and :math:`\tau` is the inverse of the variance known as the precision.
       In this model, the parameter variables :math:`\mu,\tau` are
       unobserved, so we would use variational Bayes to approximate the
       posterior distribution :math:`q(\mu, \tau) \approx p(\mu, \tau | x_1,
       \ldots, x_N)`.

    2. **Bayesian Gaussian Mixture Model**:
       A 
       `Bayesian Gaussian mixture model <https://en.wikipedia.org/wiki/Variational_Bayesian_methods#A_more_complex_example>`__
       with :math:`K` mixture components
       and :math:`N` observations :math:`\{x_1,\ldots,x_N\}`, latent categorical variables
       :math:`\{z_1,\ldots,z_N\}`, parameters :math:`{\mu_i, \Lambda_i, \pi}`
       and hyperparameters :math:`{\mu_0, \beta_0, \nu_0, W_0, \alpha_0}`, can
       be described as such:

       .. math::


           \pi &\sim \text{Symmetric-Dirichlet}_K(\alpha_0) \\
           \Lambda_{k=1,\ldots,K} &\sim \mathcal{W}(W_0, \nu_0) \\
           \mu_{k=1,\ldots,K} &\sim \mathcal{N}(\mu_0, (\beta_0\Lambda_k)^{-1}) \\
           z_i &\sim \text{Categorical}(\pi) \\
           x_i &\sim \mathcal{N}(\mu_{z_i}, \Lambda_{z_i}^{-1})  \\
           \tag{2}

       Notes:

       * :math:`\mathcal{W}` is the `Wishart distribution <https://en.wikipedia.org/wiki/Wishart_distribution>`__, which is the generalization to multiple dimensions of the gamma distribution. It's used for the prior on the covariance matrix for our multivariate normal distribution.  It's also a conjugate prior of the precision matrix (the inverse of the covariance matrix).
       * :math:`\text{Symmetric-Dirichlet}` is a `Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution>`__ which is the conjugate prior of a categorical variable (or equivalently a multinomial distribution with a single observation).

       In this case, you would want to (ideally) find an approximation to the
       joint distribution posterior (including both parameters and latent variables):
       :math:`q(\mu_1,\ldots,\mu_K, \Lambda_1,\ldots, \Lambda_K, \pi, z_1, \ldots, z_N)`
       that approximates the true posterior in all of these latent variables and parameters.

Now that we know our problem, next thing we need to is define what it means to
be a good approximation.  In many of these cases,
the `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__ (KL divergence) 
is a good choice, which is non-symmetric measure of the difference between two
probability distributions :math:`P` and :math:`Q`.  We'll discuss this in
detail in the box below, but the setup will be :math:`P` as the true posterior
distribution, and :math:`Q` being the approximate distribution, and with a bit
of math, we want to find an iterative algorithm to compute :math:`Q`.

In the mean-field approximation (a common type of variational
Bayes), we assume that the unknown variables can be partitioned so that each
partition is independent of the others.  Using KL divergence, we can derive
mutually dependent equations (one for each partition) that define the shape of
:math:`Q`.  The resultant :math:`Q` function then usually takes on the form of
well-known distributions that we can easily analyze.  The leads to an
easy-to-compute iterative algorithm (similar to the `EM algorithm
<link://slug/the-expectation-maximization-algorithm>`__) where we use all other
previously calculated partitions to derive the current one in an iterative fashion.

To summarize, variational Bayes has these ideas:

* The Bayesian inference problem of finding a posterior on the unknown
  variables (parameters and latent variables) is hard and usually can't be
  solved analytically.
* Variational Bayes solves this problem by finding a distribution :math:`Q`
  that approximates the true posterior :math:`P`.
* It uses KL-divergence as a measure of how well our approximation fits the true posterior.
* The mean-field approximation partitions the unknown variables and assumes
  each partition is independent (a simplifying assumption).
* With some (long) derivations, we can find an algorithm that iteratively computes the
  :math:`Q` distributions for a given partition by using the previous values of
  all the other partitions.

Now that we have an overview of this process, let's see how it actually works.

.. admonition:: Kullback-Leibler Divergence

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

  To quickly summarize, `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`__
  is the average amount of information or "surprise" for a probability
  distribution [1]_.  Entropy is defined as for both discrete and continuous distributions:

  .. math::

    H(P) &:= E_P[I_P(X)] = -\sum_{i=1}^n P(i) \log(P(i)) \\
    H(P) &:= E_P[I_P(X)] = - \int_{-\infty}^{\infty} p(x)\log(p(x)) dx \\
    \tag{4}

  An intuitive way to think about entropy is the (theoretical)
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
  called the 
  `cross entropy <https://en.wikipedia.org/wiki/Cross_entropy>`__ 
  of :math:`P` and :math:`Q`.  Of course, it would
  be larger than the ideal encoding, thus we would increase the average message
  length.  In other words, we need more information (or bits) to transmit a
  message from the :math:`P` distribution using :math:`Q`'s code.

  Thus, KL divergence can be viewed as this average extra-message length we need 
  when we wrongly assume the probability distribution, using :math:`Q` instead of
  :math:`P`:

  .. math::

    D_{KL}(P||Q) &= H(P,Q) - H(P)\\
                 &= -\sum_{i=1}^n P(i) \log(Q(i)) + \sum_{i=1}^n P(i) \log(P(i)) \\
                 &= \sum_{i=1}^n P(i) \log\frac{P(i)}{Q(i)} \\
                 \tag{5}

  You can probably already see how this is a useful objective to try to minimize.  If
  we have some theoretic minimal distribution :math:`P`, we want to try to find an
  approximation :math:`Q` that tries to get as close as possible by minimizing
  the KL divergence.

  |h3| Forward and Reverse KL Divergence |h3e|

  One thing to note about KL divergence is that it's not symmetric, that is,
  :math:`D_{KL}(P||Q) \neq D_{KL}(Q||P)`.  The former is called forward KL divergence,
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
  is small, we want :math:`Q` to be (proportionally) small too or the ratio
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
  Reverse KL also has the nice tendency to make our :math:`Q` distribution
  matches at least one of the modes of :math:`P`, which is really the best we
  could hope for with the shape of our approximation.

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
likelihood (sometimes called the evidence). But what if we didn't have to
directly compute the marginal likelihood and instead only needed the likelihood
(and prior)?  

This idea leads us to the two commonly used methods to solve Bayesian inference
problems: MCMC and variational inference.  You can check out my previous post on 
`MCMC <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__
but in general
it's quite slow since it involves repeated sampling but your approximation can
get arbitrarily close to the actual distribution (given enough time).
Variational inference on the other hand is a strict approximation that is much
faster because it is an optimizing problem.  It also can quantify the lower
bound on the marginal likelihood, which can help with model selection.

Now going back to our problem, we want to find an approximate distribution
:math:`Q` that minimizes the (reverse) KL divergence.  Starting from reverse KL
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

Where we're using Bayes theorem on the second line and the RHS integral
simplifies because it's simply integrating over the support of :math:`q(\theta)`
(:math:`\log p(X)` is not a function of :math:`\theta` so it factors out).
Rearranging we get:

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
variational free energy.  Notice that we don't need to compute the marginal
likelihood either, this is a big win because the likelihood and prior are usually
easily specified with the marginal likelihood intractable.  
Note that we need to find a *function*, not just a
point, that maximizes :math:`\mathcal{L}`, which means we need to use
variational calculus (see my `previous post <link://slug/the-calculus-of-variations>`__ 
on the subject), hence the name "variational Bayes".

|h2| The Mean-Field Approximation |h2e|

Before we try to derive the functional form of our :math:`Q` functions, let's 
just explicitly state some of our notation because it's going to get a bit confusing.
In the previous section, I used :math:`\theta` to represent the unknown variables.
In general, we can have :math:`N` unknown variables so 
:math:`\theta = (\theta_1, \ldots, \theta_N)` and Equation 8 and 9 will have
multiple integrals (or summations for discrete variables), one for each
:math:`\theta_i`.  I'll use :math:`\theta` to represent 
:math:`\theta_1, \ldots, \theta_N` where it is clear just to reduce the verbosity
and explicitly write it out when we want to do something special with it.

Okay, so now that's cleared up, let's move on to the mean-field approximation.
The approximation is a simplifying assumption for our :math:`Q` distribution,
which partitions the variables into independent parts (I'm just going to show
one variable per partition but you can have as many per partition as you want):

.. math::

    p(\theta|X) \approx q(\theta) = q(\theta_1, \ldots, \theta_n) = \prod_{i=1}^N q_i(\theta_i) \tag{10}


|h3| Deriving the Functional Form of :math:`q_j(\theta_j)` |h3e|

From Equation 10, we can plug it back into our variational free energy
:math:`\mathcal{L}` and try to derive the functional form of :math:`q_j` using
variational calculus [3]_.  Let's start with :math:`\mathcal{L}` and try to 
re-write it isolating the terms for :math:`q_j(\theta_j)` in hopes of taking
a functional derivative afterwards to find the optimal form of the function.
Note that :math:`\mathcal{L}` is a `functional
<https://en.wikipedia.org/wiki/Functional_(mathematics)>`_ that depends on our
approximate densities :math:`q_1, \ldots, q_N`.

.. math::

     \mathcal{L[q_1, \ldots, q_N]}
     &= - \int_{\theta_1, \ldots, \theta_N} 
        [\prod_{i=1}^N q_i(\theta_i)] \log\frac{[\prod_{k=1}^N q_k(\theta_k)]}{p(\theta,X)} 
        d\theta_1 \ldots d\theta_n \\
     &= \int_{\theta_1, \ldots, \theta_N} 
        [\prod_{i=1}^N q_i(\theta_i)] \big[
            \log p(\theta,X) - \sum_{k=1}^N \log q_k(\theta_k)
        \big]
        d\theta_1 \ldots d\theta_n \\
     &= \int_{\theta_j} q_j(\theta_j)
        \int_{\theta_{m | m \neq j}} 
        [\prod_{i\neq j} q_i(\theta_i)] \big[
            \log p(\theta,X) - \sum_{k=1}^N \log q_k(\theta_k)
        \big]
        d\theta_1 \ldots d\theta_n \\
     &= \int_{\theta_j} q_j(\theta_j)
        \int_{\theta_{m | m \neq j}} 
        [\prod_{i\neq j} q_i(\theta_i)] \log p(\theta,X)
        d\theta_1 \ldots d\theta_n \\
        &\phantom{=}-
        \int_{\theta_j} q_j(\theta_j)
        \int_{\theta_{m | m \neq j}} 
        [\prod_{i\neq j} q_i(\theta_i)] \sum_{k=1}^N \log q_k(\theta_k)
        d\theta_1 \ldots d\theta_n \\
    \tag{11}

where I'm using a bit of convenience notation in the integral index
(:math:`\theta_m|m\neq j`) so I don't have to write out the ":math:`...`".
So far, we've just factored out :math:`q_j(\theta_j)` and multiplied out
the inner term :math:`\log p(\theta, X) - \sum_{i=k}^N \log q_k(\theta_k)`.
In anticipation of the next part, we'll define some notation for an expectation
across all variables except :math:`j` as:

.. math:: 
    E_{m|m\neq j}[\log p(\theta,X)] = \int_{\theta_{m | m \neq j}} 
        [\prod_{i\neq j} q_i(\theta_i)] \log p(\theta,X)
        d\theta_1 \ldots, d\theta_{j-1}, d\theta_{j+1}, \ldots, d\theta_n \\
    \tag{12}

which you can see is just an expectation across all variables except for
:math:`j`.  Continuing on from Equation 11 using this expectation notation
and expanding the second term out:

.. math::

     \mathcal{L}[q_1, \ldots, q_N]
     &= \int_{\theta_j} q_j(\theta_j) E_{m|m\neq j}[\log p(\theta, X)] d\theta_j 
     \\
        &\phantom{=}-
        \int_{\theta_j} q_j(\theta_j) \log q_j(\theta_j)
        \int_{\theta_{m | m \neq j}} 
        [\prod_{i\neq j} q_i(\theta_i)] d\theta_1 \ldots d\theta_n \\
        &\phantom{=}-
        \int_{\theta_j} q_j(\theta_j) d\theta_j
        \int_{\theta_{m | m \neq j}} 
        [\prod_{i\neq j} q_i(\theta_i)] \sum_{k\neq j} \log q_k(\theta_k)
        d\theta_1 \ldots, d\theta_{j-1}, d\theta_{j+1}, \ldots, d\theta_n 
     \\
     &= \int_{\theta_j} q_j(\theta_j) E_{m|m\neq j}[\log p(\theta, X)] d\theta_j
        - \int_{\theta_j} q_j(\theta_j) \log q_j(\theta_j) d\theta_j\\
        &\phantom{=}-
        \int_{\theta_{m | m \neq j}} 
        [\prod_{i\neq j} q_i(\theta_i)] \sum_{k\neq j} \log q_k(\theta_k)
        d\theta_1 \ldots, d\theta_{j-1}, d\theta_{j+1}, \ldots, d\theta_n 
    \\
     &= \int_{\theta_j} q_j(\theta_j) \big[E_{m|m\neq j}[\log p(\theta, X)] - \log q_j(\theta_j)\big] d\theta_j \\
     &\phantom{=}- G[q_1, \ldots, q_{j-1}, q_{j+1}, \ldots, q_{N}]
     \\
    \tag{13}

where we're integrating probability density functions over their entire
support in a couple of places, which simplifies a few of the expressions to
:math:`1`.  It's a bit confusing because of all the indices but just take your
time to follow which index we're pulling out of which
summation/integral/product and you shouldn't have too much trouble (unless I
made a mistake!).  At the end, we have a functional that consists of a term
made up only of :math:`q_j(\theta_j)` and :math:`E_{m|m\neq j}[\log p(\theta, X)]`, 
and another term with all the other :math:`q_i` functions.

Putting together the `Lagrangian <https://en.wikipedia.org/wiki/Lagrange_multiplier>`__ 
for Equation 13, we get:

.. math::

    \mathcal{L}[q_1, \ldots, q_N] - \sum_{i=1}^N \lambda_i \int_{\theta_i} q_i(\theta_i) d\theta_i = 0 \\
    \tag{14}

where the terms in the summation are our usual probabilistic constraints that the
:math:`q_i(\theta_i)` functions must be probability density functions.

Taking the functional derivative of Equation 14 with respect to
:math:`q_j(\theta_j)` using the 
`Euler-Lagrange Equation <https://en.wikipedia.org/wiki/Calculus_of_variations#Euler.E2.80.93Lagrange_equation>`__, we get:

.. math::

    \frac{\delta \mathcal{L}[q_1, \ldots, q_N]}{\delta q_j(\theta)}
        &= \frac{\partial}{\partial q_j}\big[ 
            q_j(\theta_j) \big[E_{m|m\neq j}[\log p(\theta, X)] - \log q_j(\theta_j)\big]
            - \lambda_j q_j(\theta_j)
        \big] \\
    &= E_{m|m\neq j}[\log p(\theta, X)] - \log q_j(\theta_j) - 1 - \lambda_j \\
    \tag{15}

In this case, the functional derivative is just the partial derivative with
respect to :math:`q_j(\theta_j)` of what's "inside" the integral.  Setting to 0 and
solving for the form of :math:`q_j(\theta_j)`:

.. math::

    \log q_j(\theta_j) &= E_{m|m\neq j}[\log p(\theta, X)] - 1 - \lambda_j \\
                       &= E_{m|m\neq j}[\log p(\theta, X)] + \text{const} \\
    q_j(\theta_j) &= \frac{e^{E_{m|m\neq j}[\log p(\theta, X)]}}{Z_j} \\
    \tag{16}

where :math:`Z_j` is a normalization constant.  The constant isn't too important
because we know that :math:`q_j(\theta_j)` is a density so usually we can figure
it out after the fact. 

Equation 16 finally gives us the functional form (actually a template of the
functional form).  What usually ends up happening is that after plugging in
:math:`E_{m|m\neq j}[\log p(\theta, X)]`, the form of Equation 16 matches a
familiar distribution (e.g. Normal, Gamma etc.), and the normalization
constant :math:`Z` can be derived by inspection.  We'll see this play out in
the next section.

Taking a step back, let's see how this helps us accomplish our goal.
Recall, we wanted to maximize our variational free energy :math:`\mathcal{L}`
(Equation 9), which in turn finds a :math:`q(\theta)` that minimizes KL
divergence to the true posterior :math:`p(\theta|X)`.
Using the mean-field approximation, we broke up :math:`q(\theta)` (Equation 10) into
partitions :math:`q_j(\theta_j)`, each of which is defined by Equation 16.

However, the :math:`q_j(\theta_j)`'s are interdependent when minimizing them.
That is, to compute the optimal :math:`q_j(\theta_j)`, we need to know the
values of all the other :math:`q_i(\theta_i)` functions (because of the 
expectation :math:`E_{m|m\neq j}[\log p(\theta, X)]`).  This suggests an
iterative optimization algorithm:

1. Start with some random values for each of the parameters of the
   :math:`q_j(\theta_j)` functions.
2. For each :math:`q_j`, use Equation 16 to minimize the overall KL divergence
   by updating :math:`q_j(\theta_j)` (holding all the others constant).
3. Repeat until convergence.

Notice that in each iteration, we are lowering the KL divergence between our
:math:`Q` and :math:`P` distributions, so we're guaranteed to be improving each
time.  Of course in general we won't converge to a global maximum but it's a
heck of a lot easier to compute than MCMC.

|h2| Mean-Field Approximation for the Univariate Gaussian |h2e|

Now that we have a theoretical understanding of how this all works, let's
see it in action.  Perhaps the simplest case (and I'm using the word "simple"
in relative terms here) is the univariate Gaussian with a Gaussian prior
on its mean and a inverse Gamma prior on its variance (from Example 1).  Let's
describe the setup:

.. math::

   \mu &\sim N(\mu_0, (\kappa_0 \tau)^{-1}) \\
   \tau &\sim \text{Gamma}(a_0, b_0) \\
   X={x_1, \ldots, x_N} &\sim N(\mu, \tau^{-1}) \\
   \tag{17}

where :math:`\tau` is the precision (inverse of variance), and we have
:math:`N` observations (:math:`{x_1, \ldots, x_N}`).  For this particular
problem, there is a closed form for the posterior: a 
`Normal-gamma distribution <https://en.wikipedia.org/wiki/Normal-gamma_distribution>`__.
This means that it doesn't really make sense to compute a mean-field approximation
for any reason except pedagogy but that's why we're here right?


Continuing on, we really only need the logarithm of the joint probability of
all the variables, which is:

.. math::

    \log p(X, \mu, \tau) &= \log p(X|\mu, \tau) + \log p(\mu|\tau) + \log p(\tau) \\
    &= \frac{N}{2} \log \tau - \frac{\tau}{2} \sum_{i=1}^N (x_i - \mu)^2 \\
    &\phantom{=}
      + \frac{1}{2}\log(\kappa_0 \tau) - \frac{\kappa_0 \tau}{2}(\mu - \mu_0)^2 \\
    &\phantom{=}
      + (a_0 -1) \log \tau - b_0 \tau + \text{const}
    \tag{18}

I broke out each of the three parts into three lines, so you should be able to
easily see how we derived each of the expressions (Normal, Normal, Gamma,
respectively).  We also just absorbed all the constants into the :math:`\text{const}` term.

|h3| The Approximation |h3e|

Now onto our mean-field approximation:

.. math::

    p(\mu, \tau | X) \approx q(\mu, \tau) := q_{\mu}(\mu)q_{\tau}(\tau) \\
    \tag{19}

Continuing on, we can use Equation 16 to find the form of our :math:`q` densities.
Starting with :math:`q_{\mu}(\mu)`:

.. math::

    \log q_{\mu}(\mu) &= E_{\tau}[p(X, \mu, \tau)] + \text{const}_1 \\
      &= E_{\tau}[\log p(X|\mu, \tau) + \log p(\mu|\tau) + \log p(\tau)] + \text{const}_1 \\
      &= E_{\tau}[\log p(X|\mu, \tau) + \log p(\mu|\tau)] + \text{const}_2 \\
      &= E_{\tau}\big[\frac{N}{2} \log \tau - \frac{\tau}{2} \sum_{i=1}^N (x_i - \mu)^2
      + \frac{1}{2}\log(\kappa_0 \tau) - \frac{\kappa_0 \tau}{2}(\mu - \mu_0)^2 \big]
      + \text{const}_3 \\
      &= -\frac{E_{\tau}[\tau]}{2} \big[ \kappa(\mu - \mu_0)^2 + \sum_{i=1}^N (x_i - \mu)^2 \big] + \text{const}_4
    \tag{20}

where we absorb all terms not involving :math:`\mu` into the "const" terms
(even terms involving only :math:`\tau` because it doesn't change with respect to :math:`\mu`).
You'll notice that Equation 20 is a quadratic function in :math:`\mu`, implying
that it's normally distributed, i.e. :math:`q_{\mu}(\mu) \sim N(\mu|\mu_N, \tau_N^{-1})`.
By completing the square (or using the formula for the 
`sum of two normal distributions <https://en.wikipedia.org/wiki/Normal_distribution#Sum_of_two_quadratics>`_), we will find an expression like:

.. math::

    \log q_{\mu}(\mu) &= -\frac{(\kappa_0 + N)E_{\tau}[\tau]}{2}
                  \big( 
                    \mu - \frac{\kappa_0\mu_0 + \sum_{i=1}^N x_i}{\kappa_0 + N}
                  \big)^2 + \text{const}_5 \\
    \mu_N &= \frac{\kappa_0\mu_0 + \sum_{i=1}^N x_i}{\kappa_0 + N} \\
    \tau_N &= (\kappa_0 + N)E_{\tau}[\tau] \\
    \tag{21}

Once we completed the square in Equation 21, we can infer the mean and
precision without having to compute all those constants (thank goodness!).
Next, we can do the same with :math:`\tau`:

.. math::

    \log q_{\tau}(\tau) &= E_{\mu}[\log p(X|\tau, \mu) + \log p(\mu|\tau) + \log p(\tau)] + \text{const}_6 \\
      &= E_{\mu}\big[\frac{N}{2} \log \tau - \frac{\tau}{2} \sum_{i=1}^N (x_i - \mu)^2 \\
    &\phantom{=}
      + \frac{1}{2}\log(\kappa_0 \tau) - \frac{\kappa_0 \tau}{2}(\mu - \mu_0)^2 \\
    &\phantom{=}
      + (a_0 -1) \log \tau - b_0 \tau\big] + \text{const}_7 \\
      &= (a_0 - 1)\log \tau - b_0\tau + \frac{1}{2}\log \tau + \frac{N}{2} \log \tau \\
    &\phantom{=} -\frac{\tau}{2}E_{\mu}\big[\kappa_0(\mu - \mu_0)^2 + \sum_{i=1}^N (x_i - \mu)^2\big] + \text{const}_8 \\
    \tag{22}

We can recognize this as a :math:`\text{Gamma}(\tau|a_N, b_N)`  
because the log density is only a function of :math:`\log\tau` and
:math:`\tau`.  By inspection (and some grouping), we can find the 
parameters of this Gamma distribution (:math:`a_N, b_N`):

.. math::

    \log q_{\tau}(\tau) &= 
        \big(a_0 + \frac{N + 1}{2} - 1\big)\log\tau \\
    &\phantom{=} - \big(b_0 + \frac{1}{2}E_{\mu}\big[\kappa_0(\mu - \mu_0)^2 + \sum_{i=1}^N (x_i - \mu)^2\big] \big)\tau + \text{const}_9 \\
    a_N &= a_0 + \frac{N + 1}{2} \\
    b_N &= b_0 + \frac{1}{2}E_{\mu}\big[\kappa_0(\mu - \mu_0)^2 + \sum_{i=1}^N (x_i - \mu)^2\big] \\
    \tag{23}

Again, we don't have to explicitly compute all the constants which is really nice.
Since we know the form of each distribution, the expectation for each of the 
distributions, :math:`q(\mu) = N(\mu|\mu_N, \tau_N^{-1})` and 
:math:`q(\tau) = \text{Gamma}(\tau|a_N, b_N)`, is simple:

.. math::
    
    E_{q_{\mu}(\mu)}[\mu] &= \mu_N \\
    E_{q_{\mu}(\mu)}[\mu^2] &= \frac{1}{\tau_N} + \mu_N^2 \\
    E_{q_{\tau}(\tau)}[\tau] &= \frac{a_N}{b_N} \\
    \tag{24}

Expanding out Equations 21 and 23 to get our actual update equations:

.. math::

    \mu_N &= \frac{\kappa_0\mu_0 + \sum_{i=1}^N x_i}{\kappa_0 + N} \\
    \tau_N &= (\kappa_0 + N)\frac{a_N}{b_N} \\
    a_N &= a_0 + \frac{N + 1}{2} \\
    b_N &= b_0 + \frac{\kappa_0}{2}\big(
        E_{q_{\mu}(\mu)}[\mu^2] + \mu_0^2 - 2E_{q_{\mu}(\mu)}[\mu]\mu_0
    \big) \\
    &\phantom{=}
    + \frac{1}{2} \sum_{i=1}^N \big( x_i^2 + E_{q_{\mu}(\mu)}[\mu^2] - 2E_{q_{\mu}(\mu)}[\mu]x_i \big) \\
    \tag{25}

where in the :math:`b_N` equations, I didn't substitute some of the values 
from Equation 24 to keep it a bit neater.  From this, we can develop a simple
algorithm to compute :math:`q(\mu)` and :math:`q(\tau)`:

1. Compute values :math:`E_{q_{\mu}(\mu)}[\mu], E_{q_{\mu}(\mu)}[\mu^2], E_{q_{\tau}(\tau)}[\tau]` from Equation 24 as well as :math:`\mu_N, a_N` since they can be computed directly from the data and constants.
2. Initialize :math:`\tau_N` to some arbitrary value.
3. Use current value of :math:`\tau_N` and values from Step 1 to compute :math:`b_N`.
4. Use current value of :math:`b_N` and values from Step 1 to compute :math:`\tau_N`.
5. Repeat the last two steps until neither value has changed much.

Once we have the parameters for :math:`q(\mu)` and :math:`q(\tau)`, we can compute
anything we want such as the mean, variance, 95% credible interval etc.

|h2| Variational Bayes EM for mixtures of Gaussians [4]_ |h2e|

The previous example of a univariate Gaussian already seems a bit complex 
(one of the downsides for VB) so I just want to mention that we can do this
for the second case in Example 1 too, the Bayesian Gaussian Mixture Model.
This application of variational Bayes takes a very similar form to the 
`Expectation-Maximization <link://slug/the-expectation-maximization-algorithm>`__ 
algorithm.

Recall a mixture model has two types of variables: the latent categorical
variables for each data point specifying which Gaussian it came from (:math:`z_i`),
and the parameters to the Gaussians (:math:`\mu_k, \lambda_k`).
In variational Bayes, we treat all variables the same (i.e. find a
distribution for them), while in the EM case we only explicitly model the
uncertainty of the latent variables (:math:`z_i`) and find point estimates of
the parameters (:math:`\mu_k, \lambda_k`).
Although not ideal, the EM algorithm's assumptions are not too bad because
the parameter point-estimates use all the data points, which provides a
more robust estimate, while the latent variables :math:`z_i` are informed
only by :math:`x_i`, so it makes more sense to have a distribution. 

In any case, we still want to use variational Bayes for a mixture model situation
to allow for a more "Bayesian" analysis.  Using variational Bayes on a mixture model
produces an algorithm that is commonly known as *variational Bayes EM*.
The main idea it to just apply a mean-field approximation and factorize all
latent variables (:math:`{\bf z}`) and parameters (:math:`{\bf \theta}`):

.. math::

    p({\bf \theta}, {\bf z} | X) \approx q(\theta) \prod_1^N q(z_i) \tag{26}

Recall that the full likelihood function and prior are:

.. math::

    p({\bf z}, {\bf X} | {\bf \theta}) &= 
        \prod_i \prod_k \pi_k^{z_{ik}} \mathcal{N}(x_i | \mu_k, \Lambda_k^{-1})^{z_{ik}} \\
    p({\bf \theta}) &= 
        \text{Dir}(\pi | \alpha_0) \prod_k \mathcal{N}(\mu_0, (\beta_0\Lambda_k)^{-1}) \mathcal{W}(\Lambda_k | W_0, \nu_0) \\
    \tag{27}

We can use the same approach that we took above for both :math:`q(\theta)` and :math:`q(z_i)`
and get to a posterior of the form:

.. math::

    q({\bf z}, {\bf \theta}) &= q({\bf z})q({\theta}) \\
    &= \big[ \prod_i \text{Cat}(z_i|r_i) \big]
        \big[ \text{Dir}(\pi|\alpha) \prod_k \mathcal{N}(\mu_k|m_k, (\beta_k \Lambda_k)^{-1}W(\Lambda_k|L_k, \nu_k) \big] \\
        \tag{28}

where :math:`r_i` is the "responsibility" of a point to the clusters similar to
the EM algorithm and :math:`m_k, \beta_k, L_k, \nu_k` are computed values of the data
and hyperparameters.  I won't go into all the math because this post is getting really long and you
can just refer to Murphy or 
`Wikipedia <https://en.wikipedia.org/wiki/Variational_Bayesian_methods#A_more_complex_example>`__ 
if you really want to dig into it.

In the end, we'll end up with a two step iterative process EM-like process:

1. A variational "E" step where we compute the values latent variables (or more directly
   the responsibility) based upon the current parameter estimates of the
   mixture components.
2. A variational "M" step where we estimate the parameters of the distributions
   for each mixture component based upon the values of all the latent variables.

|h2| Conclusion |h2e|

Variational Bayesian inference is one of the most interesting topics that I have
come across so far because it marries the beauty of Bayesian inference
with the practicality of machine learning.  In future posts, I'll be exploring
this theme a bit more and start moving into techniques in the machine
learning domain but with strong roots in probability.

|h2| Further Reading |h2e|

* Previous Posts: `Variational Calculus <link://slug/the-calculus-of-variations>`__, `Expectation-Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__, `Normal Approximation to the Posterior <link://slug/the-expectation-maximization-algorithm>`__, `Markov Chain Monte Carlo Methods, Rejection Sampling and the Metropolis-Hastings Algorithm <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__, `Maximum Entropy Distributions <link://slug/maximum-entropy-distributions>`__
* Wikipedia: `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__
* Machine Learning: A Probabilistic Perspective, Kevin P. Murphy
* `A Beginner's Guide to Variational Methods: Mean-Field Approximation <http://blog.evjang.com/2016/08/variational-bayes.html>`__, Eric Jang.

|br|

.. [1] There are a few different ways to intuitively understand information entropy.  See my previous post on `Maximum Entropy Distributions <link://slug/maximum-entropy-distributions>`__ for a slightly different explanation.

.. [2] The term variational free energy is from an alternative interpretation from physics.  As with a lot of ML techniques, this one has its roots in physics where they make great use of probability to model the physical world. 

.. [3] This is one of the parts that I struggled with because many texts skip over this part (probably because it needs variational calculus).  They very rarely show the derivation of the functional form.

.. [4] This section heavily draws upon the treatment from Murphy's Machine Learning: A Probabilistic Perspective.  You should take a look at it for a more thorough treatment.
