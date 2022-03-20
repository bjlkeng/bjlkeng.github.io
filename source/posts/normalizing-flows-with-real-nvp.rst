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
normalizing flows have been one of the big ideas in deep probabilistic generative models
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

Deep probabilistic generative models, by and large, are trying to draw samples, :math:`x`
from a high dimensional distribution (e.g. images) specified :math:`p_X(X)`.
There are multiple ways to go about it but a common one used in `VAEs <link://slug/variational-autoencoders>`__ 
is a `latent variable model <https://en.wikipedia.org/wiki/Latent_variable_model>`__.
Roughly, it's a statistical model where variables are partitioned into a set of observed variables 
(:math:`x`) and latent (or hidden) ones (:math:`z`).  By assuming: 

a. Some prior :math:`p_Z(z)` (usually Gaussian) on the latent variables;
b. Some high capacity neural network :math:`g(\cdot; \theta)` (a deterministic
   function) with parameters :math:`\theta`;
c. A output distribution :math:`p_X(x|g_(z; \theta))` whose parameters are
   defined by the outputs of the neural network (e.g. :math:`g(\cdot)` defining
   the mean, variance of a normal distribution);

We can sample from our target distribution by sampling :math:`z` from our
prior, passing it through our neural network to define the parameters of our
output distribution, at which point we can finally sample from our output distribution.

This is all well and good but training our neural network to faithfully match our output
distribution is non-trivial.  Variational autoencoders are a clever way to do this by
approximating the posterior :math:`q_Z(z|x)` using another deep net, which we
simultaneously train with our latent variable model. 
See my post on `VAEs <link://slug/variational-autoencoders>`__ for more details.

One downside of this approach is that we use an approximate posterior.  This is
to get around the fact that there's not efficient method to directly train the
deep net of our latent variable model (thus build our latent variable model).

* How can we get around this?
* tantelize idea of normalizing flows... "wouldn't it be nice..."


Background
==========


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


Real NVP
========

Experiments
===========


Conclusion
==========

Further Reading
===============

* Previous posts: 
* Wikipedia: `Latent Variable Model <https://en.wikipedia.org/wiki/Latent_variable_model>`__
* [1] Dinh, Sohl-Dickstein, Bengio, Density Estimation using Real NVP, `arXiv:1605.08803 <https://arxiv.org/abs/1605.08803>`__, 2016
