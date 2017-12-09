.. title: Variational Autoencoders with Inverse Autoregressive Flows
.. slug: variational-autoencoders-with-inverse-autoregressive-flows
.. date: 2017-12-04 07:47:38 UTC-05:00
.. tags: variational calculus, autoencoders, Kullback-Leibler, generative models, MNIST, autoregressive, MADE, mathjax
.. category: 
.. link: 
.. description: An introduction to normalizing flows and inverse autoregressive flows for variational inference.
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

In this post, I'm going to be describing a really cool idea about how
to improve variational autoencoders using inverse autoregressive
flows.  The main idea is that we can generate more powerful posterior
distributions compared to the vanilla isotropic Gaussian by applying a
series of invertible transformations.  This, in theory, will allow
your variational autoencoder to fit better by concentrating the
stochastic samples around a closer approximation to the true
posterior.  The math works out so nicely while the results are kind of
marginal [1]_.  As usual, I'll go through some intuition, some math,
and have an implementation with few experiments I ran.  Enjoy!

.. TEASER_END

|h2| Motivation |h2e|

Recall in vanilla variational autoencoders (previous `post
<link://slug/variational-autoencoders>`__), we have the generative
network ("the decoder"), the posterior network ("the encoder"), which
are related to each other through a set of diagonal Gaussian variables
usually labelled :math:`Z`.

In most cases, our primary objective is to train the decoder i.e. the
generative model.  The structure of the decoder is shown in Figure 1.
We have our latent diagonal Gaussian variables, followed by a
deterministic high-capacity model (i.e. deep net), which then outputs
the parameters for our output variables (e.g. Bernoulli's if we're
modelling black and which pixels).

.. figure:: /images/variational_autoencoder4.png
  :height: 300px
  :alt: Variational "Decoder" Diagram
  :align: center

  Figure 1: The generative model component of a variational autoencoder

The main problem here is that training this network is pretty
difficult.  The brute force way is to make an input/output dataset by
taking each output you have (e.g. images) and cross it with a ton of
random samples from :math:`Z`.  If :math:`Z` has many dimensions, then
to properly cover the space you'll need an exponential number of
examples.  For example, if your number of dimensions is :math:`D=1`,
you might need to sample roughly :math:`(10)^1=10` points per image,
making your dataset 10x.  If :math:`D=10`, you'll probably need
:math:`(10)^10` points per image, making your dataset too big to
practically fit.  Not only that, most of these points you sample
will not contribute much to training your network because they'll be
in parts of the latent space that are very low probability.

Of course that's exactly why variational autoencoders are so
brilliant.  Instead of randomly sampling from your :math:`Z` space,
we'll try to use "directed" sampling that are much more likely to
occur for a given point :math:`X` using our encoder or posterior
network.  Given any data point (:math:`X`), the posterior network
generates likely :math:`Z` points to allow you to train your generator
network.  The cool thing is that we are actually training both
networks simultaneously!  To actually accomplish this though,
we have to use fully factorized diagonal Gaussian variables and a
"reparameterization trick" in order to properly get the thing to
actually work (see previous `post
<link://slug/variational-autoencoders>`__ for details).  The structure
is shown in Figure 2.

.. figure:: /images/variational_autoencoder3.png
  :height: 400px
  :alt: Variational Autoencoder Diagram
  :align: center

  Figure 2: A variational autoencoder with the "reparameterization trick".

As you can imagine, the posterior network is an estimate of the true
posterior (as is the case for variational inference methods).
Unfortunately our factorized diagonal Gaussians can't model every
distribution.  In particular, the fact that they're factorized can
limit the ability to match the true posterior we're trying to model.
Theoretically, if we are able to more closely approximate the true
posterior, our generator network should be able to train more easily
and thus improve our overall result.  

The question is how can we use a more complex distribution?  The
reason we're stuck with factorized Gaussians is because it is
(a) computationally efficient to compute and differentiate the
posterior, and (b) it's easy to sample at every minibatch.  If we
want to replace it we're going to need to still maintain these to
desirable properties.

|h2| Normalizing Flows for Variational Inference |h2e|

Normalizing flows in the context of variational inference was
introduced by Rezende in [1].  At its core, it's just applying an
invertible transformation (i.e.
`a change of variables <https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables>`__)
to our fully factorized posterior distribution to make it into
something more flexible that can match the true posterior. 
It's called a normalizing flow because the density "flows" through
each transform.  See the box below about transforming probability
density functions.

.. admonition:: Transforming Probability Density Functions

    Given a n-dimensional random variables :math:`\bf X` with joint
    density function :math:`f({\bf x})`, we can transform it into
    another n-dimensional random variable :math:`\bf y` via a
    invertible (i.e. 1-to-1) and differentiable function :math:`H`
    with joint density :math:`g({\bf y})`:


    .. math::

        {\bf y} = H({\bf X}) \tag{1}

    It turns out the joint density :math:`g({\bf y})` can be computed
    as:

    .. math::

        g({\bf y}) &= f({\bf x})\big|\text{det}(\frac{d{\bf x}}{d{\bf y}})\big| \\
                   &= f(H^{-1}({\bf y}))\big|\text{det}(\frac{d{H^{-1}({\bf y})}}{d{\bf y}})\big|
        \tag{2}

    where the second part there is the 
    `determinant of the Jacobian <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Jacobian_determinant>`__ 
    of the inverse :math:`H^{-1}` at :math:`{\bf y}`.
    In the last line we're making it explicit that the density is a
    function :math:`\bf y`.  Alternatively, we can also write this in
    terms of :math:`{\bf x}` and :math:`H`:

    .. math::

        g({\bf y}) &= f({\bf x})\big|\text{det}(\frac{d{\bf x}}{d{\bf y}})\big| \\
                   &= f({\bf x})\big|\text{det}(\big[\frac{dH({\bf x})}{d{\bf x}}\big]^{-1})\big| \\
                   &= f({\bf x})\big|\text{det}(\frac{dH({\bf x})}{d{\bf x}})\big|^{-1} 
                   \tag{3}

    It's a bit confusing because of all the variable changing but keep
    in mind that you can change between :math:`{\bf x}` with :math:`{\bf y}` without
    much trouble because there is a 1-1 mapping.  
    So starting from the original statement, we first apply the
    `inverse function theorem <https://en.wikipedia.org/wiki/Inverse_function_theorem>`__,
    which allows us to re-write the Jacobian matrix in terms of
    :math:`H` and :math:`{\bf x}`.  Next, we apply a 
    `property <https://en.wikipedia.org/wiki/Determinant#Multiplicativity_and_matrix_groups>`__ 
    of determinants which says the determinant of an inverse of a
    matrix is just the reciprocal of the deteminant of the original
    matrix.

    It's a bit strange why we would want to put things back in terms
    of :math:`{\bf x}` and :math:`H` but sometimes (like in this post)
    we want to evaluate :math:`g({\bf y})` but don't want to
    explicitly compute :math:`H^{-1}`. Instead, it's easier to just
    work with :math:`H`.

Let's see some math to make this idea a bit more precise.
Start off with a simple posterior for our initial variable, call it
:math:`\bf z_0`.  In vanilla VAEs we use a fully factorized Gaussian
e.g.  :math:`q({\bf z_0}|{\bf x}) \sim \mathcal{N}({\bf \mu(x)}, {\bf \sigma(x)})`.
Next, we'll want to apply a series of invertible transforms
:math:`f_t(\cdot)`:

.. math::

    {\bf z_0} &\sim q({\bf z_0}|{\bf x}) \\
    {\bf z_t} &\sim f_t({\bf z_{t-1}}, {\bf x}) & \forall t=1..T
    \tag {4}

Remember :math:`{\bf x}` parameterizes our posterior distribution
(i.e. our encoder), so we can freely use it as part of our
transformation here.
Using Equation 3, we can compute the (log) density of the final
variable :math:`{\bf z_t}` as such:

.. math::

    \log q({\bf z_T} | {\bf x}) = \log q({\bf z_0}|{\bf x})
    - \sum_{t=1}^T \log \big| det(\frac{d{\bf z_t}}{d{\bf z_{t-1}}}) \big|
      \tag{5}

Equation 5 is simply just a repeated application of Equation 3,
noticing that since we're in log space the reciprocal of the
determinant turns into a negative sign.

So why does this help at all?  The normalizing flow can be thought of
as a sequence of expansions or contractions on the initial density,
allowing for things such as multi-modal distributions (something we
can't do with a basic Gaussian).  Additionally, it allows us to have
complex relationships between the variables instead of the
independence we assume with our diagonal Gaussians.
So then the trick then is to pick a transformation :math:`f_t` that
gives us the flexibility but, importantly, is easy to compute
because we want to use this in a VAE setup.  The next section
describes an elegant and simple to compute transform that accomplishes
both of these things.

|h2| Inverse Autoregressive Transformations |h2e|



.. figure:: /images/iaf.png
  :height: 400px
  :alt: Autoregressive Transform
  :align: center

  Figure 3: Autoregressive Transform and Inverse Autoregressive
  Transform


* Autoregressive property: `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__, [2]
* Autoregressive equations
* Show derivation of inverse autoregressive equations (the Gaussian form)
* Diagram showing forward and backwards

|h2| Inverse Autoregressive Flows |h2e|

* Show block diagram from paper [3]
  * Context 'h'
* Jacobians are triangular wrt to dmu/dz_{t-1}
* Show a VAE diagram
* Explain stable computation sigma * z + (1-sigma) * m

|h3| Experiments: IAF Implementation |h3e|

TODO

|h3| Implementation Notes |h3e|

* The "context" vector is connected to the input of the MADE with additional dense layer (see impl.)
* I added a 2 * to the stability computation, seems like you will need it or else you can't "expand" the volume
* Had a lot of trouble with getting things to be stable with made computation, not sure if all of them helped:
    * Added regularizers on the autoregressive parts
    * Used 'sigmoid' for activation for all made stuff (instead of 'elu' for others)
* Had a bunch of confusion with the logqz_x computation, in particular the determinant.  Only after I worked through the math did I actually figure out the sign of the determinant in Eq. X
* 

|h2| Conclusion |h2e|

|h2| Further Reading |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, `Semi-supervised Learning with Variational Autoencoders <link://slug/semi-supervised-learning-with-variational-autoencoders>`__, `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__
* My implementation on Github:
* [1] "Variational Inference with Normalizing Flows", Danilo Jimenez Rezende, Shakir Mohamed, `ICML 2015 <https://arxiv.org/abs/1505.05770>`__
* [2] "MADE: Masked Autoencoder for Distribution Estimation", Germain, Gregor, Murray, Larochelle, `ICML 2015 <https://arxiv.org/pdf/1502.03509.pdf>`__
* [3] "Improving Variational Inference with Inverse Autoregressive Flow", Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling, `NIPS 2016 <https://arxiv.org/abs/1606.04934>`_
* Wikipedia: `Probability Density Function: Dependent variables and change of variables <https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables>`__
* Github code for "Improving Variational Inference with Inverse Autoregressive Flow": https://github.com/openai/iaf/

.. [1] At least by my estimate the results are kind of marginal. Improving the posterior on its own doesn't seem to have a significant boost in the likelihood.  The IAF paper [3] actually does have really good results on CIFAR10 but uses a novel architecture combined with IAF transforms.  So by itself, the IAF doesn't do that much.
