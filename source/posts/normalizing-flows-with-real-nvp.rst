.. title: Normalizing Flows with Real NVP
.. slug: normalizing-flows-with-real-nvp
.. date: 2022-03-18 13:36:05 UTC-04:00
.. tags: normalizing flows, generative models, CIFAR10, CELEBA, MNIST, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

Write your post here.

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
* Wikipedia: 
* [1] Dinh, Sohl-Dickstein, Bengio, Density Estimation using Real NVP, `arXiv:1605.08803 <https://arxiv.org/abs/1605.08803>`__, 2016
