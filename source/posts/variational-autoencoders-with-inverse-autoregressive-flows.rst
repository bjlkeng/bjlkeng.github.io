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

An autoregressive transform is one where given a sequence of variables 
:math:`{\bf y} = \{y_i \}_{i=0}^D`, each variable is dependent only on the
previously indexed variables i.e. :math:`y_i = f_i(y_{0:i-1})`.

Autoregressive autoencoders introduced in [2] 
(and my `post on it <link://slug/autoregressive-autoencoders>`__)
take advantage of this property by constructing an extension of vanilla
autoencoder that can estimate distributions (whereas the regular one doesn't
have a direct probabilistic interpretation).  The paper introduced the idea in
terms of binary Bernoulli variables, but we can also formulate it in terms of
Gaussians too.

When looking at defining the :math:`f_i(\cdot)` function, you only need a
single function to estimate the parameter :math:`p` for a Bernoulli.  For a
Gaussian, we'll need two to estimate the mean and variance denoted by
:math:`[\bf \mu(y), \sigma(y)]`.  However, due to the autoregressive property,
the individual elements are only functions of the prior indices 
and thus their derivatives are zero with respect to latter indices:

.. math::
    \mu_i &= f_i(y_{0:i-1})\\
    \sigma_i &= g_i(y_{0:i-1}) \\
    \frac{\partial[\mu_i, \sigma_i]}{\partial y_j} &= [0, 0] &\text{ for } j\geq i \\
    \tag{6}

Given :math:`[\bf \mu(y), \sigma(y)]`, we can sample from the autoencoder
by sequentially transforming a noise vector 
:math:`\epsilon \sim \mathcal{N}(0, {\bf I})` as such:

.. math::
    y_0 &= \mu_0 + \sigma_0 \odot \epsilon_. \\
    y_i &= \mu_i({\bf y_{0:i-1}}) + \sigma_i({\bf y_{0:i-1}}) \odot \epsilon_0 & \text{ for } i > 0
    \tag{7}

where addition is element-wise and :math:`\odot` is element-wise multiplication.

However in our case, we can transform any vector, not just a noise vector.
This is shown on the left hand side of Figure 3 where we take a :math:`\bf x`
vector and transform it to a :math:`\bf y` vector.  You'll notice that we have
to perform :math:`D` sequential computations, thus making it too slow for our
intended purpose of use in a normalizing flow.  But what about the inverse
transform?

.. figure:: /images/iaf.png
  :width: 550px
  :alt: Autoregressive Transform
  :align: center

  Figure 3: Gaussian version of Autoregressive Transform and Inverse
  Autoregressive Transforms

The inverse transform can actually be parallelized (we'll switch to :math:`\bf x` as
the input variable instead of :math:`\epsilon`) as shown in Equation 8 and
the right hand side of Figure 3 so long as we have :math:`\sigma_i > 0`.

.. math::

    x_i = \frac{y_i - \mu_i({\bf y_{0:i-1}})}{\sigma_i({\bf y_{0:i-1}})} \tag{8}

where subtraction and division is element-wise.

Now here comes the beauty of this transform.  Recall in normalizing flows
to properly compute the probability, we have to be able to compute
the determinant (or log-determinant).  However, Equation 6 shows that
:math:`\bf \mu, \sigma` have no dependence on the current or latter variables
in our sequence.  Thus from Equation 8, the Jacobian is lower triangular with a
simple diagonal:

.. math::

    \frac{d{\bf x}}{d{\bf y}} =
    \begin{bmatrix}
    \frac{1}{\sigma_0} & 0 & \dots & 0 \\
    \frac{\partial x_0}{\partial y_1} & \frac{1}{\sigma_1} & \dots  & 0 \\
    \vdots & \ddots & \ddots & \vdots \\
    \frac{\partial x_0}{\partial y_D} & \dots & \frac{\partial x_{D-1}}{\partial y_D} &  \frac{1}{\sigma_D} \\
    \end{bmatrix}
    \tag{9}

Knowing that the determinant of a triangular matrix is the product of its
diagonals, this gives us our final result for the log determinant:

.. math::

    \log \big| det \frac{d{\bf x}}{d{\bf y}} \big|
    = - \sum_{i=0}^D \log \sigma_i({\bf y}_{0:i-1}) \tag{10}
    
In the next section, we'll see how to use this inverse autoregressive transform
to build a more flexible posterior distribution for our variational
autoencoder.

|h2| Inverse Autoregressive Flows |h2e|

Adding an inverse autoregressive flow to a variational autoencoder is as
simple as (a) adding a bunch of IAF transforms after the latent variables
:math:`z` (b) modifying the likelihood to account for the IAF transforms.

Figure 4 from [3] shows a depiction of add IAF transforms to a variational
encoder.
Two things to note: a context :math:`h` is additionally generated and the IAF
step only involves multiplication and addition, not division and subtraction
like in Equation 8.  I'll explain these two points as we work through the math
below.

.. figure:: /images/iaf2.png
  :alt: Autoregressive Transform
  :align: center

  Figure 4: (Left) An Inverse Autoregressive Flow (IAF) transforming the basic
  posterior of an variational encoder to a more complex posterior through
  multiple IAF transforms. (Right) A single IAF transform. (Source: [3])


To start off, we generate our basic latent variables as we would in an
vanilla autoencoder starting from a standard diagonal Gaussian
:math:`{\bf \epsilon} \sim \mathcal{N}(0, I)`:

.. math::

    {\bf z}_0 = {\bf \mu}_0 + {\bf \sigma}_0\odot {\bf \epsilon} \tag{11}

where :math:`{\bf \mu}_0, {\bf \sigma}_0` are generated by our encoder network.
So far nothing too exciting.

Now here's the interesting part, we want to apply Equation 8 as an IAF transform
to go from :math:`{\bf z}_t` to :math:`{\bf z}_{t+1}` (indices refer to the
number of transforms applied on the vector :math:`\bf z`, not the index into
:math:`z`) but there are a few issues we need to resolve first.  Let's start
with reinterpreting Equation 8, rewriting it in terms of :math:`{\bf z}_t` and
:math:`{\bf z}_{t+1}` (we'll omit the indices into the vector with the
understanding that :math:`\mu` and :math:`\sigma` are autoregressive):

.. math::

    {\bf z}_{t+1}
        &= \frac{{\bf z}_t - {\bf \mu}_t({\bf z}_t)}{{\bf \sigma}_t({\bf z}_t)} \\
        &= \frac{{\bf z}_t}{{\bf \sigma}_t({\bf z}_t)} -
           \frac{{\bf \mu}_t({\bf z}_t)}{{\bf \sigma}_t({\bf z}_t)} \\
        &= {\bf z}_t \odot {\bf s}_t({\bf z}_t) -
           {\bf m}_t({\bf z}_t) \\
    \tag{12}

where :math:`{\bf s}_t = \frac{1}{{\bf \sigma}_t}` and 
:math:`{\bf m}_t = \frac{{\bf \mu}_t}{{\bf \sigma}_t}`.  We can do this re-writing
because remember that we are learning these functions through a neural network 
so it doesn't really matter if invert or negate.


    
where the first subscript is the number of steps we've applied so far in our
chain of IAF transforms, and the second indexes the vector of latent variables.


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
