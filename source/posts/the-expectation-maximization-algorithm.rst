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
<https://en.wikipedia.org/wiki/Graphical_model>`_.  

One of the simplest kinds of PGMs is when you have a 1-1 mapping between your
latent variables (usually represented by :math:`z_i`) and observed variables
(:math:`x_i`), and your latent variables take on discrete values (:math:`z_i
\in {1,\ldots,K}`).  We'll be focusing on this much simpler
case as explained in the next section.

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

Visually, we can imagine the density of the observed
variables (housing prices) as the "sum" or mixture of several Gaussians (image
from http://dirichletprocess.weebly.com/clustering.html): 

.. image:: /images/gmm.png
   :height: 300px
   :alt: Latent Variables
   :align: center

So when a value is observed, there is an implicit latent variable that decided
which of the Gaussians (neighbourhoods) it came from.  

Following along with this housing price example, let's represent the price of
each house as real-valued random variable :math:`x_i` and the unobserved
neighbourhood it belongs to as a discrete valued random variable :math:`z_i` [2]_.
Further, let's suppose we have :math:`K` neighbourhoods therefore,
:math:`z_i` can be modelled as a
`categorical distribution <https://en.wikipedia.org/wiki/Categorical_distribution>`_
with parameter :math:`\pi = [\pi_1, \ldots, \pi_k]`, and the price distribution
of the :math:`k^{th}` neighbourhood as a Gaussian :math:`\mathcal{N}(\mu_k,
\sigma_k^2)` with mean :math:`\mu_k` and variance :math:`\sigma_k^2`.  
The density, then, of :math:`x_i` is given by:

.. math::

    p(x_i|\theta) &=  \sum_{k=1}^K p(z_i=k) p(x_i| z_i=k, \mu_k, \sigma_k^2)  \\
    x_i| z_i &\sim \mathcal{N}(\mu_k, \sigma_k^2) \\
    z_i &\sim \text{Categorical}(\pi) \tag{1}
    
Where :math:`\theta` represents the parameters of the Gaussians (all the :math:`\mu_k,
\sigma_k^2`) and the categorical variables (:math:`\pi`).  Notice that since
:math:`z_i` variables are non-observed, we need to `marginalize
<https://en.wikipedia.org/wiki/Marginal_distribution>`_ them out to get the
density of the observed variables (:math:`x_i`).  Translating Equation 1 to
plainer language: we model the price distribution of each house as a linear
combination [3]_ ("mixture model") of our :math:`K` Gaussians (neighbourhoods).

Now we have a couple of relevant inference problems, given different
assumptions:

1. Assuming you know the values of all the parameters (:math:`\theta`), compute
   the *responsibility*, :math:`r_{ik}`, of a cluster :math:`k` to a
   point :math:`i`: :math:`r_{ik} = p(z_i=k | x_i, \theta)`.

   This essentially tells you how "likely" or "close" a point is to an existing
   cluster.  We'll use this below in the EM algorithm but this computation can
   also be used for GMM classifiers to find out which class :math:`x_i` belongs
   to.

2. Estimating the parameters of the Gaussians (:math:`\mu_k, \sigma^2`) and categorical
   variable (:math:`\pi`) given a) Just the observed points (:math:`x_i`);
   b) The observed points (:math:`x_i`) **and** the values of the latent variables
   (:math:`z_i`).
    
   The former problem is the general unsupervised learning problem that we'll solve
   with the EM algorithm (e.g. finding the neighbourhoods).  The latter is a
   specific problem that we'll indirectly use as one of the steps in the EM
   algorithm.  Coincidently, this latter problem is the same one when using
   GMMs for classification except we label the :math:`z_i` as :math:`y_i`.

We'll cover the steps needed to compute both of these in the next section.

|h2| The Expectation-Maximization Algorithm |h2e|

The `Expectation-Maximization (EM) Algorithm
<https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_ is
an iterative method to find the MLE or MAP estimate for models with latent
variables.  This is a description of how the algorithm works from 10,000 feet:

0. **Initialization**: Get an initial estimate for parameters
   :math:`\theta^0` (e.g. all the :math:`\mu_k, \sigma_k^2` and :math:`\pi`
   variables).  In many cases, this can just be a random initialization.
1. **Expectation Step**: Assume the parameters (:math:`\theta^{t-1}`) from the
   previous step are fixed, compute the expected values of the latent variables
   (or more often a *function* of the expected values of the latent variables).
2. **Maximization Step**: Given the values you computed in the last step 
   (essentially known values for the latent varibles), estimate new values
   for :math:`\theta^t` that maximize a variant of the likelihood function.
3. **Exit Condition**: If likelihood has not changed much, exit; otherwise, go
   back to Step 1.

One very nice part about Steps 2 and 3 are that they are quite easy to compute
separately because we're not trying to figure out both the latent variables and
the model parameters at the same time.  It turns out that every iteration of
the algorithm will increase the likelihood function, implying a better fit.
However, the likelihood function is non-convex so we're only guaranteed to
approach a local maxima.  One way to get around this by running the algorithm
for multiple initial values to get broader coverage of the parameter space.

|h3| EM algorithm for Gaussian Mixture Models |h3e|

Coming back to GMMs, let's review what information we have when we're
estimating them (i.e. problem 2(a) from the previous section).
To start, we have a bunch of observed variables (:math:`x_i`).  Since
we've decided on using a GMM model, we also have to pick the hyper parameter
:math:`K` that decides how many Gaussians we want in our model [4]_.
That's about all the information we have.  Given that, the next algorithm
(using pseudo-Python) describes how we would estimate the relevant unknowns:

.. code:: python
    
    # Assume we have function to compute density of Gaussian 
    # at point x_i given mu, sigma: G(x_i, mu, sigma); and
    # a function to compute the log-likelihoods: L(x, mu, sigma, pi)
    def estimate_gmm(x, K, tol=0.001, max_iter=100):
        ''' Estimate GMM parameters.
            :param x: list of observed real-valued variables
            :param K: integer for number of Gaussian
            :param tol: tolerated change for log-likelihood
            :return: mu, sigma, pi parameters
        '''
        # 0. Initialize theta = (mu, sigma, pi)
        N = len(x)
        mu, sigma = [rand()] * K, [rand()] * K
        pi = [rand()] * N
        
        current_L = np.inf
        for j in range(max_iter):
            prev_L = curr_L
            # 1. E-step: responsibility = p(z_i = k | x_i, theta^(t-1))
            r = {}
            for i in range(N):
                parts = [pi[k] * G(x_i, mu[k], sigma[k]) for i in range(K)]
                total = sum(parts)
                for i in k:
                    r[(i, k)] = parts[k] / total

            # 2. M-step: Update mu, sigma, pi values
            rk = [sum([r[(i, k)] for i in range(N)]) for k in range(K)]
            for k in range(K):
                pi[k] = rk[k] / N
                mu[k] = sum(r[(i, k)] * x[i] for i in range(N)) / rk[k]
                sigma[k] = sum(r[(i, k)] * (x[i] - mu[k]) ** 2) / rk[k]

            # 3. Check exit condition
            curr_L = L(x, mu, sigma, pi)
            if abs(prev_L - curr_L) < tol:
                break

        return mu, sigma, pi

Caution: This is just an illustration of the algorithm, please don't use it!
It probably suffers from a lot of real-world issues like floating point
overflow.  However, we can still learn something from it.  Let's break the
major computation steps down to understand the math behind it.

In the Expectation Step, we assume that we know the values of all the parameters
(:math:`\theta = (\mu_k, \sigma_k^2, \pi)`) are fixed and are set to the ones
from the previous iteration of the algorithm.  We then just need to compute the
responsibility of each cluster to each point.  Re-phasing this problem:
Assuming you know the locations of each of the :math:`K` Gaussians
(:math:`\mu_k, \sigma_k`), and the overall distribution of the latent variables
(:math:`pi_k`), what is the probability that a given point :math:`x_i` is drawn
from cluster :math:`k`?

We can write this in terms of probability and use Bayes theorem to find the
answer:

.. math::

    r_{ik} = p(z_i=k|x_i, \theta) &= \frac{p(x_i | z_i=k, \theta) \cdot p(z_i=k)}{\sum_{j=1}^K p(x_i | z_i=j, \theta) \cdot p(z_i=j)} \\
    &= \frac{\mathcal{N}(x_i | \mu_k, \sigma_k) \cdot \pi_k}
            {\sum_{j=1}^K \mathcal{N}(x_i | \mu_j, \sigma_j) \cdot \pi_j}
    \tag{2}

This is just the normalized probability of each each point belonging to one of
the :math:`K` Gaussians weighted by the mixture distribution (:math:`\pi_k`).
We'll see later on that this expression actually comes out by taking an
expectation over the complete data log likelihood function, which is where the
"E" comes from.  In any case, this step becomes quite simple once we can assume
that the parameters :math:`\theta` are fixed.

The Maxmization Step turns things around and assumes the responsibilities
(proxies for the latent variables) are fixed, and now the problem is we want to
maximize our (expected complete data log) likelihood function across all the
:math:`\theta = (\mu_k, \sigma_k^2, \pi)` variables.  We'll show the math of
how to arrive at these expressions below and describe the intuitive
interpretation here.

First up, the overall distribution of the latent variables :math:`\pi`.
Assuming you know all the values of the latent variables (i.e. :math:`r_{ik}`:
how much each point :math:`x_i` contributes to each cluster :math:`k`), then
intuitively, we just need to sum up the contribution to each cluster and
normalize (just like we would estimate the distibution of a six-sided dice
roll):

.. math::

    \pi_k = \frac{1}{N} \sum_i r_{ik} \tag{3}

Next, we need to estimate the Gaussians.  Again, since we know the
responsibilities of each point to each cluster, we can just use our standard
methods for estimating the mean and standard deviation of Gaussians but
weighted according to the responsibilities:

.. math::

    \mu_k &= \frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}} \\
    \sigma_k &= \frac{\sum_i r_{ik}(x_i - \mu_k)(x_i - \mu_k)}{\sum_i r_{ik}} \tag{4}

Again we shall see that this comes out from the expected complete data log
likelihood function.

As a last note, there are many variants of this algorithm.  The most popular being the
`K-Means algorithm <https://en.wikipedia.org/wiki/K-means_clustering>`_.  In
this variant, we assume that both the shape of the Gaussians
(:math:`\sigma_k = \sigma^2I_D`) and distribution of latent variables
:math:`\pi=\frac{1}{K}` are fixed, so now all we have to compute are the
cluster centers.  The other big difference is that we now perform *hard
clustering*, where we assign responsibility of a point :math:`x_i` to exactly
one cluster (and zero responsibility to other clusters).  These assumptions
simplify Equation 2-4 while keeping all the nice properties of the EM algorithm,
making it quite a popular algorithm for unsupervised clustering.

|h2| Expectation-Maximization Math |h2e|

In this section, we'll go over some of the derivations and proofs related to
the EM algorithm.  It's going to get a bit math-heavy but that's usually where
I find that I get the best intuition.

|h3| Complete Data Log-Likelihood and the :math:`Q` function |h3e|

Recall the overall goal of the EM algorithm is finding an MLE (or MAP)
estimation in a model with unobserved latent variables.  MLE estimates
by definition attempt to maximize the likelihood function.  In the 
general case, with observations :math:`x_i` and latent variables :math:`z_i`,
we have the log-likelihood as:

.. math::

    l(\theta) = \sum_{i=1}^N \log p(x_i|\theta)
              = \sum_{i=1}^N \log \sum_{z_i} p(x_i, z_i | \theta) \tag{5}

The first expression is just the plain definition of the likelihood function
(the probability that the data fits a given set of a parameters).  The second
expression shows that we need to marginalize out (integrate out if it were
continuous) the unobserved latent variable :math:`z_i`.
Unfortunately, this expression is hard to optimize because we can't push
the "log" inside the summation.  The EM algorithm gets around this by
defining a related quantity called the *complete data log-likelihood* function:

.. math::

    l_c(\theta) = \sum_{i=1}^N \log p(x_i, z_i | \theta) \tag{6}

Again this cannot be computed because we don't know :math:`z_i` but now
we can take the expectation of Equation 6 with respect to our unobserved
variables :math:`z_i`.  Additionally, we introduce the idea that
:math:`z_i` is a fixed function of some

**TODO CONTINUE ON** 

.. math::

    Q(\theta, \theta^{t-1}) = E[l_c(\theta) | D, \theta^{t-1}] \tag{7}

The notation might be a bit confusing but let's break it down.  The
first thing to notice is that we have the concept of iterations now
with the introduction of the expectation.  The expectation is actually
taken over


The other question you may have is why are we defining this 
:math:`Q(\theta, \theta^{t-1})` function?  It turns out that improving the
:math:`Q` function will never cause a loss in our actual likelihood function,
we'll show this down below.  Therefore, the EM loop should always improve
our likelihood function (up to a local maximum).


|h3| EM for Gaussian Mixture Models |h3e|




* 

|h3| Proof of Correctness for EM ** |h3e|


|h2| Conclusion |h2e|


|h2| Further Reading |h2e|

* Wikipedia: 
  `Expectation-Maximization algorithm <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_,
  `Mixture Models <https://en.wikipedia.org/wiki/Mixture_model>`_
* Machine Learning: A Probabilistic Perspective, Kevin P. Murphy

|br|

.. [1] The material in this post is heavily based upon the treatment in *Machine Learning: A Probabilistic Perspective* by Kevin P.  Murphy; it has a much more detailed explanation and I encourage you to check it out.

.. [2] This is actually only one application of Gaussian Mixture Models.  Another common one is using it as a generative classifier i.e. estimating :math:`p(X_i, y_i)` (where we label :math:`z_i` as :math:`y_i` as per convention for classifiers).  Since both :math:`X_i` and :math:`y_i` are observable, it's much easier to directly estimate the density versus the case where we have to infer values for hidden variables.

.. [3] I say linear combination because we don't actually know the value of :math:`z_i`, so one way to think about it is the expected value of :math:`z_i`.  This translates to :math:`x_i` having a portion of each of the :math:`K` Gaussians being responsible for generating it.  Thus, the linear combination idea.

.. [4] Picking :math:`K` is non-trivial since for the typical application of unsupervised learning, you don't know how many clusters you have! Ideally, some domain knowledge will help drive that decision or more often than not you vary :math:`K` until the results are useful for your application.
