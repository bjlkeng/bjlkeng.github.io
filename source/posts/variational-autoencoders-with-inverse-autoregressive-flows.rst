.. title: Variational Autoencoders with Inverse Autoregressive Flows
.. slug: variational-autoencoders-with-inverse-autoregressive-flows
.. date: 2017-12-19 08:47:38 UTC-05:00
.. tags: variational calculus, autoencoders, Kullback-Leibler, generative models, MNIST, autoregressive, MADE, CIFAR10, mathjax
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
distributions compared to a more basic isotropic Gaussian by applying a
series of invertible transformations.  This, in theory, will allow
your variational autoencoder to fit better by concentrating the
stochastic samples around a closer approximation to the true
posterior.  The math works out so nicely while the results are kind of
marginal [1]_.  As usual, I'll go through some intuition, some math,
and have an implementation with few experiments I ran.  Enjoy!

.. TEASER_END

|h2| Motivation |h2e|

Recall in vanilla variational autoencoders (`previous post
<link://slug/variational-autoencoders>`__), we have the generative
network ("the decoder"), the posterior network ("the encoder"), which
are related to each other through a set of diagonal Gaussian variables
usually labelled :math:`z`.

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

The main problem here is that training this generative model is pretty
difficult.  The brute force way is to make an input/output dataset by
taking each output you have (e.g. images) and cross it with a ton of
random samples from :math:`z`.  If :math:`z` has many dimensions, then
to properly cover the space you'll need an exponential number of
examples.  For example, if your number of dimensions is :math:`D=1`,
you might need to sample roughly :math:`(10)^1=10` points per image,
making your dataset 10x.  If :math:`D=10`, you'll probably need
:math:`(10)^{10}` points per image, making your dataset too big to
practically train.  Not only that, most the sampled points will not contribute
much to training your network because they'll be in parts of the latent space
that are very low probability (with respect to the current image), contributing
almost nothing to your network weights.

Of course that's exactly why variational autoencoders are so
brilliant.  Instead of randomly sampling from your :math:`z` space,
we'll use "directed" sampling to a pick point :math:`z` using our encoder (or
posterior) network such that :math:`x` is much more likely to occur.  
This allows you to efficiently train your generator network.  The cool thing is
that we are actually training both networks simultaneously!  To actually
accomplish this though, we have to use fully factorized diagonal Gaussian
variables and a "reparameterization trick" in order to properly get the thing
to actually work (see my `previous post <link://slug/variational-autoencoders>`__
for details).  The structure is shown in Figure 2.

.. figure:: /images/autoencoder_reparam_trick.png
  :height: 300px
  :alt: Variational Autoencoder Diagram
  :align: center

  Figure 2: Left: A naive implementation of an autoencoder without the
  reparameterization trick.  Right: A vanilla variational autoencoder with the
  "reparameterization trick" (Source: [4])

As you can imagine, the posterior network is an estimate of the true
posterior (as is the case for variational inference methods).
Unfortunately our fully factorized diagonal Gaussians can't model every
distribution.  In particular, the fact that they're fully factorized can
limit the ability to match the true posterior we're trying to model.
Theoretically if we are able to more closely approximate the true
posterior, our generator network should be able to train more easily,
and thus improve our overall result.  

The question is how can we use a more complex distribution?  The
reason we're stuck with factorized Gaussians is because it is
(a) computationally efficient to compute and differentiate the
posterior (just backprop through the :math:`z` mean and variance), and (b) it's
easy to sample at every minibatch (which is just sampling from independent
Gaussians because of the reparameterization trick).  If we want to replace it
we're going to need to still maintain these two desirable properties.

|h2| Normalizing Flows for Variational Inference |h2e|

Normalizing flows in the context of variational inference was
introduced by Rezende et al. in [1].  At its core, it's just applying an
invertible transformation (i.e.
`a change of variables <https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables>`__)
to our fully factorized posterior distribution to make it into
something more flexible that can match the true posterior. 
It's called a normalizing flow because the density "flows" through
each transform.  See the box below about transforming probability
density functions.

.. admonition:: Transforming Probability Density Functions

    Given a n-dimensional random variable :math:`\bf x` with joint
    density function :math:`f({\bf x})`, we can transform it into
    another n-dimensional random variable :math:`\bf y` via an
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

    where the latter part of each line contains a  
    `determinant of a Jacobian <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Jacobian_determinant>`__.
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
    matrix is just the reciprocal of the determinant of the original
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

Remember our posterior distribution (i.e. our encoder) is conditioned
on :math:`{\bf x}`, so we can freely use it as part of our transformation here.
Using Equation 3, we can compute the (log) density of the final variable
:math:`{\bf z_t}` as such:

.. math::

    \log q({\bf z_T} | {\bf x}) = \log q({\bf z_0}|{\bf x})
    - \sum_{t=1}^T \log \big| \text{det}(\frac{d{\bf z_t}}{d{\bf z_{t-1}}}) \big|
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
The trick then is to pick a transformation :math:`f_t` that
gives us the flexibility, but more importantly is easy to compute
because we want to use this in a VAE setup.  The next section
describes an elegant and simple to compute transform that accomplishes
both of these things.

|h2| Inverse Autoregressive Transformations |h2e|

An autoregressive transform is one where given a sequence of variables 
:math:`{\bf y} = \{y_i \}_{i=0}^D`, each variable is dependent only on the
previously indexed variables i.e. :math:`y_i = f_i(y_{0:i-1})`.

Autoregressive autoencoders introduced in [2] 
(and my `post on it <link://slug/autoregressive-autoencoders>`__)
take advantage of this property by constructing an extension of a vanilla
(non-variational) autoencoder that can estimate distributions (whereas the regular one doesn't
have a direct probabilistic interpretation).  The paper introduced the idea in
terms of binary Bernoulli variables, but we can also formulate it in terms of
Gaussians too.

Recall that Bernoulli variables only have a single parameter :math:`p` to
estimate, but for a Gaussian we'll need two functions to estimate the mean and
variance denoted by :math:`[{\bf \mu}({\bf y}), {\bf \sigma}({\bf y})]`.
However, due to the autoregressive property, the individual elements are only
functions of the prior indices and thus their derivatives are zero with respect
to latter indices:

.. math::
    \mu_i &= f_i(y_{0:i-1})\\
    \sigma_i &= g_i(y_{0:i-1}) \\
    \frac{\partial[\mu_i, \sigma_i]}{\partial y_j} &= [0, 0] &\text{ for } j\geq i \\
    \tag{6}

Given :math:`[{\bf \mu}({\bf y}), {\bf \sigma}({\bf y})]`, we can
sequentially apply an autoregressive transform on a noise vector 
:math:`\epsilon \sim \mathcal{N}(0, {\bf I})` as such:

.. math::
    y_0 &= \mu_0 + \sigma_0 \odot \epsilon_0. \\
    y_i &= \mu_i({\bf y_{0:i-1}}) + \sigma_i({\bf y_{0:i-1}}) \odot \epsilon_i & \text{ for } i > 0
    \tag{7}

where addition is element-wise and :math:`\odot` is element-wise multiplication.
Intuitively, this is a natural way to make Gaussians, you multiply by some
standard deviation and add some mean.

However in our case, we can transform any vector, not just a noise vector.
This is shown on the left hand side of Figure 3 where we take a :math:`\bf x`
vector and transform it to a :math:`\bf y` vector.  You'll notice that we have
to perform :math:`\mathcal{O}(D)` sequential computations, thus making it too slow for our
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
diagonals, this gives us our final result for the log determinant which
is incredibly simple to compute:

.. math::

    \log \big| det \frac{d{\bf x}}{d{\bf y}} \big|
    = - \sum_{i=0}^D \log \sigma_i({\bf y}_{0:i-1}) \tag{10}
    
In the next section, we'll see how to use this inverse autoregressive transform
to build a more flexible posterior distribution for our variational
autoencoder.

|h2| Inverse Autoregressive Flows |h2e|

Adding an inverse autoregressive flow (IAF) to a variational autoencoder is as
simple as (a) adding a bunch of IAF transforms after the latent variables
:math:`z` (b) modifying the likelihood to account for the IAF transforms.

Figure 4 from [3] shows a depiction of adding several IAF transforms to a
variational encoder.
Two things to note: (1) a context :math:`h` is additionally generated, and
(2) the IAF step only involves multiplication and addition, not division and
subtraction like in Equation 8.  I'll explain these two points as we work
through the math below.

.. figure:: /images/iaf2.png
  :alt: Autoregressive Transform
  :align: center

  Figure 4: (Left) An Inverse Autoregressive Flow (IAF) transforming the basic
  posterior of an variational encoder to a more complex posterior through
  multiple IAF transforms. (Right) A single IAF transform. (Source: [3])


To start off, we generate our basic latent variables as we would in an
vanilla VAE starting from a standard diagonal Gaussian
:math:`{\bf \epsilon} \sim \mathcal{N}(0, I)`:

.. math::

    {\bf z}_0 = {\bf \mu}_0 + {\bf \sigma}_0\odot {\bf \epsilon} \tag{11}

where :math:`{\bf \mu}_0, {\bf \sigma}_0` are generated by our encoder network.
So far nothing too exciting.

Now here's the interesting part, we want to apply Equation 8 as an IAF transform
to go from :math:`{\bf z}_t` to :math:`{\bf z}_{t+1}` (indices refer to the
number of transforms applied on the vector :math:`\bf z`, not the index into
:math:`\bf z`) but there are a few issues we need to resolve first.  Let's start
with reinterpreting Equation 8, rewriting it in terms of :math:`{\bf z}_t` and
:math:`{\bf z}_{t+1}` (we'll omit the indices into the vector with the
understanding that :math:`\mu` and :math:`\sigma` are autoregressive):

.. math::

    {\bf z}_{t+1}
        &= \frac{{\bf z}_t - {\bf \mu}_t({\bf z}_t)}{{\bf \sigma}_t({\bf z}_t)} \\
        &= \frac{{\bf z}_t}{{\bf \sigma}_t({\bf z}_t)} -
           \frac{{\bf \mu}_t({\bf z}_t)}{{\bf \sigma}_t({\bf z}_t)} \\
        &= {\bf z}_t \odot {\bf s}_t({\bf z}_t) +
           {\bf m}_t({\bf z}_t) \\
    \tag{12}

where :math:`{\bf s}_t = \frac{1}{{\bf \sigma}_t}` and 
:math:`{\bf m}_t = -\frac{{\bf \mu}_t}{{\bf \sigma}_t}`.  We can do this re-writing
because remember that we are learning these functions through a neural network 
so it doesn't really matter if we invert or negate.

Next, let's introduce a context :math:`\bf h`.  Recall, our posterior is 
:math:`p({\bf z}|{\bf x})` but this is the same as just saying
:math:`p({\bf z}|{\bf x}, f({\bf x}))`, where :math:`f(\cdot)` is some
deterministic function.  So let's just define :math:`{\bf h}:=f({\bf x})`
and then use it in our IAF transforms.  This is not a problem because our
latent variable is still only a function of :math:`{\bf x}`:

.. math::

    {\bf z}_{t+1}
        &= {\bf z}_t \odot {\bf s}_t({\bf z}_t, {\bf h}) + {\bf m}_t({\bf z}_t, {\bf h}) \\
    \tag{13} 

Equation 13 now matches Figure 4 (where we've relabelled :math:`\mu, \sigma` to
:math:`m, s` for clarity).

The last thing to note is that in the actual implementation, [3] suggests
a modification to improve numerical stability while fitting.
Given the outputs of the autoregressive network:

.. math::

    [{\bf m}_t, {\bf s}_t] = \text{AutoregressiveNN}[t]({\bf z}_t, {\bf h}; {\bf \theta})
    \tag{14}

We construct :math:`{\bf z}_t` as:

.. math::

    {\bf z}_t = \text{sigm}({\bf s}_t)\odot {\bf z}_{t-1} + (1 - \text{sigm}({\bf s}_t))\odot {\bf m}_t
    \tag{15}

where sigm is the sigmoid function.  This is inspired by an LSTM-style
updating.  They also suggest to initialize the weights of :math:`{\bf s}_t` to
saturate the sigmoid so that it's mostly just a pass-through to start.
During experimentation, I saw that if I didn't use this LSTM-style trick, by the
third or fourth IAF transform I started getting NaNs really quickly.


|h3| Deriving the IAF Density |h3e|

Now that we have Equation 15, we can finally derive the posterior density. 
Even with the change of variable above, :math:`{\bf s}, {\bf m}`  are still autoregressive
so Equation 10 applies.  Using this fact, starting from Equation 5:

.. math::

    \log q({\bf z_T} | {\bf x}) &= \log q({\bf z_0}|{\bf x})
    - \sum_{t=1}^T \log \big| det(\frac{d{\bf z_t}}{d{\bf z_{t-1}}}) \big| \\
    &= \log q({\bf z_0}|{\bf x}) - \sum_{t=1}^T \big[- \sum_{i=0}^D \log {\bf \sigma}_{t, i}\big] & \text{Equation 10} \\
    &= \log q({\bf z_0}|{\bf x}) - \sum_{t=1}^T \big[\sum_{i=0}^D \log {\bf s}_{t, i}\big] & \text{Change variable to }{\bf s} \\
    &= - \sum_{i=0}^D \big[ \frac{1}{2}\epsilon_i^2 + \frac{1}{2}\log(2\pi) + \sum_{t=0}^D \log {\bf s}_{t, i}\big]  \\
      \tag{16}

where :math:`q({\bf z}_0|{\bf x})` is just an isotropic Gaussian centered at
:math:`{\bf \mu}_0` with :math:`{\bf \sigma}_0`.  Here we apply Equation 3 
to transform back to an expression involving just :math:`\bf \epsilon`, and
absorb the :math:`{\sigma}_0` into the summation (by changing variable to
:math:`{\bf s}`).

Lastly, we can write our entire variational objective as:

.. math::

  \log{p({\bf x})} &\geq -E_q\big[\log\frac{q({\bf z}_T|{\bf x})}{p({\bf z}_T,{\bf x})}\big]  \\
             &= E_q\big[\log p({\bf z}_T,{\bf x}) - \log q({\bf z}_T|{\bf x})\big] \\
             &= E_q\big[\log p({\bf x}|{\bf z}_T) + \log p({\bf z}_T) - \log q({\bf z}_T|{\bf x})\big] \\
    \log q({\bf z_T} | {\bf x})
    &= - \sum_{i=0}^D \big[ \frac{1}{2}\epsilon_i^2 + \frac{1}{2}\log(2\pi) + \sum_{t=0}^D \log {\bf s}_{t, i}\big]  \\
    \log p({\bf z_T}) &= - \sum_{i=0}^D \big[ \frac{1}{2}{\bf z}_T^2 + \frac{1}{2}\log(2\pi)\big]  \\
    \tag{17}

with :math:`\log p({\bf z_T})` as our standard diagonal Gaussian prior and whatever
distribution you want on your output variable :math:`\log p({\bf x}|{\bf z}_T)` 
(e.g. Bernoulli, Gaussian, etc.).
With Equation 17, you can just negate it and use it as your loss function for
your IAF VAE (the expectation gets estimated implicitly with the mini-batches
of your stochastic gradient descent).

|h2| Experiments: IAF Implementation |h2e|

I implemented a VAE with IAF on both a binarized MNIST dataset as well as CIFAR10
in a set of `notebooks <https://github.com/bjlkeng/sandbox/tree/master/notebooks/vae-inverse_autoregressive_flows>`__.
My implementation uses a modification of the MADE autoencoder from [2] (see my previous post on `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__)
for the IAF layers.  I only used the "basic" version of the IAF network from
the paper, not the extension based on ResNet.
As usual, things were pretty easy to put together using Keras because I
basically took the code for a VAEs, added the code I had for a MADE, and
modified the loss function as needed [2]_.

The network for the encoder consists of a few convolution layers, a couple of dense
layers, and a symmetric structure for the decoder except with transposed
convolutions.  I used 32 latent dimensions for MNIST and 128 for CIFAR10 with
proportional number of filters and hidden nodes for the convolutional and dense
layers respectively.  For the IAF portion, I used separate MADE layers (with 2
hidden layers each) for the :math:`\bf m` and :math:`\bf s` variables with 10x
hidden nodes relative to the latent dimensions.  As the paper suggested, I
reversed the order of :math:`\bf z_t` after each IAF transform.

.. csv-table:: Table 1: IAF Results
   :header: "Model", "Training Loss", "Validation Loss", "P(x|z) Test Loss"
   :widths: 15, 10, 10, 10
   :align: center

   "MNIST-VAE", 70.92, 72.33, 40.19
   "MNIST-VAE+IAF", 66.44, 70.89, 39.99
   "CIFAR10-VAE", 1815.29, 1820.17, 1781.58
   "CIFAR10-VAE+IAF", 1815.07, 1823.05, 1786.24

Table 1 shows the results of the experiments on both datasets.  As you can see
the IAF layers seems to do a bit of improvement on MNIST taking the validation
loss down from :math:`72.3` to :math:`70.9`, while there's barely an
improvement on the test output loss.  This didn't really have any affect on the
generated images, which qualitatively showed no difference.  The IAF layers
seemed improve the MNIST numbers a bit but only marginally.

The CIFAR10 results seemed to get worse on validation/test sets.  One thing
that I did notice is that adding more IAF layers requires more training
(there are many more parameters) and also you need some "tricks" in order
to properly train it.  This is likely due to the long chain of dense layers
that make up the IAF transforms, similar to troubles you might have in an LSTM.
So it's quite possible that the IAF layers could be slightly beneficial but
I just didn't train it quite right.

My overall conclusion is that IAF didn't seem to have a huge affect on the
resulting output (at least on its own).  I was hoping that it would help VAEs
achieve GAN-like results but alas, it's not quite there.  I will note that [3]
actually had a novel architecture using IAF transforms with a ResNet like
structure.  This is where they achieved near state-of-the-art results on
CIFAR10.  Probably this structure is much easier to train (like ResNet) and
really allows you to take advantage of the IAF layers.


|h3| Implementation Notes |h3e|

- The "context" vector is connected to the input of the MADE with an additional dense layer (see implementation).
- The trick from Equation 15 was needed.  Even at 4 IAF layers I started getting NaNs pretty quickly.
- Even beyond Equation 15, I had a lot of trouble with getting things to be
  stable with the MADE computations, not sure if all of them helped:

  - Added regularizers on the MADE layers.
  - Used 'sigmoid' for activation for all MADE stuff (instead of the 'elu' activation I used for the other layers).
  - I disabled dropout for all layers.

- I had a bunch of confusion with the :math:`\log q(z|x)` computation, especially
  the determinant.  Only after I worked through the math did I actually figure
  out the sign of the determinant in Equation 17.  The key is really understanding
  the change of variables in Equation 12.
- I actually spent a lot of time trying to get IAFs to work on a "toy" example
  using various synthetic data like mixed Gaussians or weird auto-regressive
  distributions.  In all these cases, I had a lot of trouble showing that the
  IAF transforms did anything, it looks to perform pretty much on par with a
  vanilla VAE.  My hypothesis on this is that either the distributions were too
  simple so the vanilla VAE works really well, or the IAF transforms don't do
  much.  I suspect the latter is probably most of it.
- Now I'm wondering if the normalizing flow transforms from [1] will do a better job
  but I didn't spend any time trying to implement it to see if it made a difference.

|h2| Conclusion |h2e|

Well there you have, another autoencoder post!  When I first read about this idea
I was super excited because conceptually it's so beautiful!  Using the exact same
VAE that we all know and love, you can improve its performance just by transforming
the posterior and removing the big perceived limitation: diagonal Gaussians.
Unfortunately, normalizing flows with IAF transforms are no silver bullet and the
improvements I saw were pretty small (if you know otherwise please let me know!).
Despite this, I still really like the idea, at least theoretically, because I think
it really shows some creativity to use the *inverse* of an autoregressive flow
in addition to the whole concept of a normalizing flow.  Who knew transforming
probability distributions would be useful?  Anyways, I learned lots of really 
interesting things working on this and you can expect more in the new year!
Happy Holidays!

|h2| Further Reading |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, `Semi-supervised Learning with Variational Autoencoders <link://slug/semi-supervised-learning-with-variational-autoencoders>`__, `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__
* My implementation on Github: `notebooks <https://github.com/bjlkeng/sandbox/tree/master/notebooks/vae-inverse_autoregressive_flows>`__
* [1] "Variational Inference with Normalizing Flows", Danilo Jimenez Rezende, Shakir Mohamed, `ICML 2015 <https://arxiv.org/abs/1505.05770>`__
* [2] "MADE: Masked Autoencoder for Distribution Estimation", Germain, Gregor, Murray, Larochelle, `ICML 2015 <https://arxiv.org/pdf/1502.03509.pdf>`__
* [3] "Improving Variational Inference with Inverse Autoregressive Flow", Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling, `NIPS 2016 <https://arxiv.org/abs/1606.04934>`_
* [4] "Tutorial on Variational Autoencoders", Carl Doersch, `<http://arxiv.org/abs/1606.05908>`__

* Wikipedia: `Probability Density Function: Dependent variables and change of variables <https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables>`__
* Github code for "Improving Variational Inference with Inverse Autoregressive Flow": https://github.com/openai/iaf/

.. [1] At least by my estimate the results are kind of marginal. Improving the posterior on its own doesn't seem to have a significant boost in the ELBO or the output variable likelihood.  The IAF paper [3] actually does have really good results on CIFAR10 but uses a novel architecture combined with IAF transforms.  But by itself, the IAF doesn't seem to do that much.
.. [2] Actually, it was a bit harder than that because I had to work through a bunch of bugs and misinterpretations of the math, but the number of lines added was quite small.
