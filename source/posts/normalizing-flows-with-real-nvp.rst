.. title: Normalizing Flows with Real NVP
.. slug: normalizing-flows-with-real-nvp
.. date: 2022-03-18 13:36:05 UTC-04:00
.. tags: normalizing flows, generative models, CIFAR10, CELEBA, MNIST, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

This post has been a long time coming.  I originally started working on it several posts back but
hit a roadblock in the implementation and then got distracted with some other ideas, which took
me down various rabbit holes (`here <link://slug/hamiltonian-monte-carlo>`__,
`here <link://slug/lossless-compression-with-asymmetric-numeral-systems>`__, and
`here <link://slug/lossless-compression-with-latent-variable-models-using-bits-back-coding>`__). 
It feels good to finally get back on track to some of the core ML topics that I was learning about.
The other nice thing about not being an academic researcher (not that I'm
really researching anything here) is that there is no pressure to do anything!
If it's just for fun, you can take your time with a topic, veer off track, and
the come back to it later.  It's nice having the freedom to do what you want (this applies to
more than just learning about ML too)!

This post is going to talk about a class of deep probabilistic generative
model called normalizing flows.  Alongside `Variational Autoencoders <link://slug/variational-autoencoders>`__
and autoregressive models (e.g. `Pixel CNN <link://slug/pixelcnn>`__ and 
`Autoregressive autoencoders <link://slug/autoregressive-autoencoders>`__), 
normalizing flows have been one of the big ideas in deep probabilistic generative models[1]_
(I don't count GANs aren't counted here because they are not quite probabilistic).
Specifically, I'll be presenting Real NVP, one of the earlier normalizing flow
techniques named *Real NVP* (circa 2016). 
The formulation is simple but surprisingly effective, which makes it a good
candidate to study to understand more about normalizing flows.
As usual, I'll go over some background, the method, an implementation 
(with commentary on the details), and some experimental results.  Let's get into the flow!

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

Given a distribution :math:`p_X(X)`, deep generative models use neural networks to model :math:`X`
usually by minimizing some quantity related to the negative log-likelhood (NLL) :math:`-\log(P(X))`.
Assuming we have identical, independently distributed (IID) samples :math:`x \in X`, we 
are aiming for a loss that is related to:

.. math::

   \sum_{x \in X} -logp_X(X) \tag{1}

There are multiple ways to build a deep generative model but a common way is to use is a 
`latent variable model <https://en.wikipedia.org/wiki/Latent_variable_model>`__,
where we partition the variables into two sets: observed variables (:math:`x`)
and latent (or hidden) ones (:math:`z`).  We only ever observe :math:`x` and
usually use the latent :math:`z` variables because they make the problem more
tractable.  We can sample from this latent variable model by having three things:

a. Some prior :math:`p_Z(z)` (usually Gaussian) on the latent variables;
b. Some high capacity neural network :math:`g(z; \theta)` (a deterministic
   function) with input :math:`z` and model parameters :math:`\theta`;
c. A conditional output distribution :math:`p_{X|Z}(x|g_(z; \theta))` whose
   distribution parameters are defined by the outputs of the neural network (e.g.
   :math:`g(z;\theta)` define the mean, variance of the assumed normal
   distribution of :math:`X`).

By sampling :math:`z` from our prior, passing it through our neural network to
define the parameters of our output distribution :math:`p_{X|Z}`, and finally defining
our target distribution :math:`p_{X|Z}`, we can finally sample a point from it.
This is all well and good but the real tricky part is training this model!
Let's see why.

We wish to minimize Equation 1 (our loss function) but we only have our
conditional distribution :math:`p_{X|Z}`.  We can get most of the way there
by using our prior :math:`p_Z`.  From Equation 1:

.. math::

   \sum_{x \in X} -\log p_X(X) &= \sum_{x \in X} -\log\big(\int_{z} p_{X,Z}(x,z) dz\big) \\
   &= \sum_{x \in X} -\log\big(\int_{z} p_{X|Z}(x|z)p_Z(z) dz\big) \\
   &\approx \sum_{x \in X} -\log\big(\sum_{i=1}^K p_{X|Z}(x|z_i)p_Z(z_i)\big) &&& \text{Approx. by using } K \text{ } z_i \in Z \text{ samples} \\
   \tag{2}

There are a couple of issues here.  First, we have this summation inside the
logarithm, that's usually a tough thing to optimize.  Perhaps the more
important issue though is that we have to draw :math:`K` samples from :math:`Z`
*for every* :math:`X`.  If we use any reasonable number of latent variables,
we immediately hit `curse of dimensionality <https://en.wikipedia.org/wiki/Curse_of_dimensionality>`__
issues with the number of samples we need.

Variational autoencoders are a clever way around this by approximating the
posterior :math:`q_Z(z|x)` using another deep net, which we simultaneously
train with our latent variable model.  Using the 
`expected lower bound objective <https://en.wikipedia.org/wiki/Evidence_lower_bound>`__ (ELBO)
we can indirectly optimize (an upper bound of) :math:`-\log P(X)`.  See my post
on `VAEs <link://slug/variational-autoencoders>`__ for more details.

This is great but can we define a deep generative model that does this more
directly?  What if we could directly optimize :math:`p_X(x)` but still had the
convenience of starting our sampling process from a simple distribution of
:math:`z` variables?  Of course we can (otherwise it would be a terrible setup)!
Read on to find out how but let's review some background material first.

Background
==========

The first two concepts we need are the
`Inverse Transform Sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`__ and
`Probability Integral Transform <https://en.wikipedia.org/wiki/Probability_integral_transform>`__.
Inverse transform sampling is idea that given a random variable :math:`X`
(under some mild assumptions) with CDF :math:`F_X`, we can sample from :math:`X` 
using starting from a standard uniform distribution :math:`U`.  This can be easily seen
by sampling :math:`U` and using the inverse CDF `F^{-1}_X` to generate a random sample 
from :math:`X`.  The probability integral transform is the opposite operation:
given a way to sample :math:`X` (and its associated CDF), we can generate a
sample from a standard uniform distribution :math:`U` as :math:`u=F_X(x)`.
See the box below for more details.

Using these two ideas (and its extension to multiple variables), there exists a
*deterministic* transformation (recall CDFs and their inverses are
deterministic functions) to go from any distribution :math:`X` to any
distribution :math:`Y`.  This can be achieved by transforming from :math:`X` to 
a standard uniform distribution :math:`U` (probability integral transform), then
going from :math:`U` to :math:`Y` (inverse transform sampling).  For our purposes,
we don't actually care to explicitly specify the CDFs but rather just understand
that this transformation from samples of :math:`X` to :math:`Y` exists via a 
*deterministic* function.  Notice that this deterministic function is *bijective*
(or invertible) because the CDFs (and inverse CDFs) are monotone functions.

.. admonition:: Inverse Transform Sampling

    `Inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`__
    is a method for sampling from any distribution given its cumulative
    distribution function (CDF), :math:`F(x)`. 
    For a given distribution with CDF :math:`F(x)`, it works as such:

    1. Sample a value, :math:`u`, between :math:`[0,1]` from a uniform
       distribution.
    2. Define the inverse of the CDF as :math:`F^{-1}(u)` (the domain is a 
       probability value between :math:`[0,1]`).
    3. :math:`F^{-1}(u)` is a sample from your target distribution.

    Of course, this method has no claims on being efficient.  For example,
    on continuous distributions, we would need to be able to find the inverse
    of the CDF (or some close approximation), which is not at all trivial.
    Typically, there are more efficient ways to perform sampling on any
    particular distribution but this provides a theoretical way to
    sample from *any* distribution.

    **Proof** 

    The proof of correctness is actually pretty simple.  Let :math:`U`
    be a uniform random variable on :math:`[0,1]`, and :math:`F^{-1}`
    as before, then we have:

    .. math::

        &P(F^{-1}(U) \leq x) \\
        &= P(U \leq F(x)) && \text{apply } F \text{ to both sides} \\
        &= F(x)  && \text{because } P(U\leq y) = y \text{ on } [0,1] \\
        \tag{3}

    Thus, we have shown that :math:`F^{-1}(U)` has the distribution
    of our target random variable (since the CDF :math:`F(x)` is the same).  
    
    It's important to note what we did: we took an easy to sample random
    variable :math:`U`, performed a *deterministic* transformation
    :math:`F^{-1}(U)` and ended up with a random variable that was distributed
    according to our target distribution.

    **Example** 

    As a simple example, we can try to generate a exponential distribution
    with CDF of :math:`F(x) = 1 - e^{-\lambda x}` for :math:`x \geq 0`.
    The inverse is defined by :math:`x = F^{-1}(u) = -\frac{1}{\lambda}\log(1-y)`.
    Thus, we can sample from an exponential distribution just by iteratively
    evaluating this expression with a uniform randomly distributed number.

    .. figure:: /images/Inverse_transformation_method_for_exponential_distribution.jpg
      :height: 300px
      :alt: Visualization of mapping between a uniform distribution and an exponential one (source: Wikipedia)
      :align: center
    
      Figure 1: The :math:`y` axis is our uniform random distribution and the :math:`x` axis is our exponentially distributed number.  You can see for each point on the :math:`y` axis, we can map it to a point on the :math:`x` axis.  Even though :math:`y` is distributed uniformly, their mapping is concentrated on values closer to :math:`0` on the :math:`x` axis, matching an exponential distribution (source: Wikipedia).

    **Extensions** 

    Now instead of starting from a uniform distribution, what happens if we
    want to sample from another distribution, say a normal distribution?
    We just first apply the reverse of the inverse sampling transform
    called the 
    `Probability Integral Transform <https://en.wikipedia.org/wiki/Probability_integral_transform>`__.
    So the steps would be:

    1. Sample from a normal distribution.
    2. Apply the probability integral transform using the CDF of a normal
       distribution to get a uniformly distributed sample.
    3. Apply inverse transform sampling with the inverse CDF of the target
       distribution to get a sample from our target distribution.

    What about extending to multiple dimensions?  We can just break up the
    joint distribution into its conditional components and sample each
    sequentially to construct the overall sample:

    .. math::

        P(x_1,\ldots, x_n) = P(x_n|x_{n-1}, \ldots,x_1)\ldots P(x_2|x_1)P(x_1) \tag{4}

    In detail, first sample :math:`x_1` using the method above, then :math:`x_2|x_1`,
    then :math:`x_3|x_2,x_1`, and so on.  Of course, this implicitly means you
    would have the CDF of each of those distributions available, which
    practically might not be possible.


The next thing we need is to review is how to `change variables of probability density functions <https://en.wikipedia.org/wiki/Probability_density_function#Densities_associated_with_multiple_variables>`__.
Given continuous n-dimensional random variable :math:`Z` with joint density :math:`p_Z`
and a bijective (i.e. invertible) differentiable function :math:`g`, let :math:`X=g(Z)`,
then :math:`p_X` is defined by:

.. math::

    p_X(x) &= p_Z(z)\big|det\big(\frac{\partial z}{\partial x}\big)\big| \\
    &= p_Z(g^{-1}(x))\big|det\big(\frac{\partial g^{-1}(x)}{\partial x}\big)\big| \\
    &= p_Z(f(x))\big|det\big(\frac{\partial f(x)}{\partial x}\big)\big| && \text{Define }f := g^{-1} \\
    \tag{5}
  
where :math:`\big|det\big(\frac{\partial f(x)}{\partial x}\big)\big|` is the 
`determinant of the Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`__.
The determinant comes into play because we're essentially changing variables of
the density function in the CDF integral.

We'll see later that using this change of variable formula with the (big)
assumption of a bijective function, we can eschew the approximate posterior (or
in the case of GANs the discriminator network) to train our deep generative model
directly.

Normalizing Flows with Real NVP
===============================

The two big ideas from the previous section come together using this simplified logic:

1. There exists an invertible transform :math:`f: X \rightarrow Z` to convert
   between any two probability densities (Inverse Transform Sampling and
   Probability Integral Transform); define a deep neural network to be this
   invertible function :math:`f`.
2. We can compute the (log-)likelihood of any variable :math:`X=f^{-1}(Z)` (for
   invertible :math:`f`) by just knowing the density of :math:`Z` and the function :math:`f`
   (i.e. not explicitly knowing the density of :math:`X`) using Equation 5.
3. Thus, we can train a deep latent variable model directly using its
   log-likelihood as a loss function with simple latent variables :math:`Z` 
   (e.g Gaussians) and an invertible deep neural network (:math:`f`) to model
   some unknown complex distribution :math:`X` (e.g. images).

Notice there are two things that we are doing that give normalizing flows [2] its namesake:

* **"Normalizing"**: The change of variable formula (Equation 5) gives us a
  normalized probability density.
* **"Flow"**: A series of invertible transforms that are composed together to
  make a more complex invertible transform.

Now the big assumption here is that you can build a deep neural network that is
both *invertible* and can represent whatever complex transform you need.  There
are several methods to do this but we'll be looking at one of the earlier ones
call Real NVP, which is surprisingly simple.

Training and Generation
-----------------------

As previously mentioned, normalizing flows greatly simplify the training process.
No need for approximate posteriors (VAEs) or discriminator networks (GANs) to 
train -- just directly minimize the negative log likelihood.  Let's take a closer look
at that.

Assume we have training samples from a complex data distribution :math:`X`, a
deep neural network :math:`z = f_\theta(x)` parameterized by `\theta`, and a prior
:math:`p_Z(z)` on latent variables :math:`Z`.   From Equation 5, we can 
derive our log-likelihood function like so:

.. math::

    \log p_X(x) &= \log\Big(p_Z(f_\theta(x))\big|det\big(\frac{\partial f_\theta(x)}{\partial x}\big)\big| \Big) \\
    &= \log p_Z(f_\theta(x)) + \log\Big(\big|det\big(\frac{\partial f_\theta(x)}{\partial x}\big)\big| \Big)
    \tag{6}

As in many of these deep generative models, if we assume a standard independent 
Gaussian priors for :math:`p_Z`, we can replace the first term in Equation 6
with the logarithm of the standard normal PDF:

.. math::

    \log p_X(x) &= \log p_Z(f_\theta(x)) + \log\Big(\big|det\big(\frac{\partial f_\theta(x)}{\partial x}\big)\big| \Big) \\
                &= -\frac{1}{2}\log(2\pi) - \frac{(f_\theta(x))^2}{2}
                + \log\Big(\big|det\big(\frac{\partial f_\theta(x)}{\partial x}\big)\big| \Big) && \text{assume Gaussian prior} \\
    \tag{7}

Thus, our training is straight forward, just do a forward pass with training
example :math:`x` and do a backwards pass using the negative of Equation 7 as
the negative log-likelihood loss function.  The tricky part is defining
a bijective deep generative model (described below) and computing the
determinant of the Jacobian.  It's not obvious how to design a expressive
bijective deep neural network while it's even less obvious how to compute its
Jacobian determinant efficiently (recall the Jacobian could be very large).
We'll cover both in the next section.

Generating samples is also quite straight forward because :math:`f_\theta` is
invertible.  Starting from a randomly sample point from our prior distribution
on :math:`Z` (e.g. standard Gaussian), we can generate a sample easily by using
the inverse of our deep net: `x = f^-1_\theta(z)`.  So a nice property of
normalizing flows is that the training and generation of samples is fast
(as opposed to autoregressive models where generation is very slow).

.. admonition:: Data Preprocessing and Compute the Density

   * Talk about pixel space
   * normalizing between [0,1]
   * the transform they use in the paper
   * Images need to take this into account when computing metrics 
   * Show equation where you add a preprocess function :math:`h(f_\theta(x))` and
     its associated Jacobian (which is usuall diagonal) if done pixel by pixel


Coupling Layers
---------------

* Masked convolutions

Stacking Coupling Layers
------------------------

Multi-Scale Architecture
------------------------

Modified Batch Normalization
----------------------------

Experiments
===========


Conclusion
==========

Further Reading
===============

* Previous posts: 
* Wikipedia: `Latent Variable Model <https://en.wikipedia.org/wiki/Latent_variable_model>`__, `Probabilify Density Function <https://en.wikipedia.org/wiki/Probability_density_function#Vector_to_vector>`__, `Inverse Transform Sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`__, `Probability Integral Transform <https://en.wikipedia.org/wiki/Probability_integral_transform>`__, `Change of Variables in the Probability Density Function <https://en.wikipedia.org/wiki/Probability_density_function#Densities_associated_with_multiple_variables>`__
* [1] Dinh, Sohl-Dickstein, Bengio, Density Estimation using Real NVP, `arXiv:1605.08803 <https://arxiv.org/abs/1605.08803>`__, 2016
* [2] Stanforrd CS236 Class Notes, `<https://deepgenerativemodels.github.io/notes/flow/>`__

.. [1] Apparently, autoregressive models can be interpreted as flow-based models (see [2]) but it's not very intuitive to me so I like to think of them as their own separate thing.
