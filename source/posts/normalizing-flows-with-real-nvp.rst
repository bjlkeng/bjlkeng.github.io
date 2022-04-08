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
Specifically, I'll be presenting one of the earlier normalizing flow
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
    
      **Figure 1: The :math:`y` axis is our uniform random distribution and the :math:`x` axis is our exponentially distributed number.  You can see for each point on the :math:`y` axis, we can map it to a point on the :math:`x` axis.  Even though :math:`y` is distributed uniformly, their mapping is concentrated on values closer to :math:`0` on the :math:`x` axis, matching an exponential distribution (source: Wikipedia).**

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
call Real-valued Non-Volume Preserving (Real NVP) transformations, which is
surprisingly simple.

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

Coupling Layers
---------------

So the key question for normalizing flows is how can you define an invertible
deep neural network?  Real NVP uses a surprisingly simple block called an
"affine coupling layer".  The main idea is to define a transform whose Jacobian
forms a triangular matrix resulting in a very simple and efficient determinant
computation.  Let's first define the transform.

The coupling layer is a simple scale and shift operation for some *subset* of
the variables in the current layer, while the other half are used to compute
the scale and shift.  Given D dimensional input variables :math:`x`,
:math:`y` as the output of the block, and :math:`d < D`:

.. math::

    y_{1:d} &= x_{1:d} \\
    y_{d+1:D} &= x_{d+1:D} \odot exp(s(x_{1:d})) + t(x_{1:d}) \\
    \tag{8}

where :math:`s` is for scale, :math:`t` is for translation, and are functions
from :math:`R^d \mapsto R^{D-d}`, and :math:`\odot` is the element wise product.
The reverse computation is just as simple by solving for :math:`x` and noting
that :math:`x_{1:d}=y_{1:d}`:

.. math::

    x_{1:d} &= y_{1:d} \\
    x_{d+1:D} &= (y_{d+1:D}  - t(y_{1:d})) \odot exp(-s(y_{1:d})) \\
    \tag{9}

.. figure:: /images/realnvp_coupling.png
  :height: 270px
  :alt: Visualization of Affine Coupling Layer
  :align: center

  **Figure 2: Forward and reverse computations of affine coupling layer [1]**

Figure 2 is a figure from [1] that shows this visually.  It's not at all obvious
(at least to me) that this simple transform can represent the complex bijections
that we want from our deep net.  However, I'll point out two ideas.  First,
:math:`s(\cdot)` and :math:`t(\cdot)` can be arbitrarily *deep* networks with
width greater than the input dimensions.  This essentially can scale and shift
the input :math:`x` in complex ways.  Second, we're going to be stacking a lot 
of these together.  So while it seems like for a subset of the variables
(:math:`x_{1:d}`) we're not doing anything, in fact, we scale and shift every
input variable multiple times.  Still, there's no proof or guarantees in the
paper that these transforms can represent every possible bijection but the
empirical results are surprisingly effective.

From our coupling layer in Equation 8, we can easily derive the Jacobian
from Equation 6:

.. math::

   \frac{\partial y}{\partial x^T} = 
   \begin{bmatrix}
       I_d       & 0 \\
       \frac{\partial y_{d+1:D}}{\partial x^T_{1:d}}      & diag(exp[s(x_{1:D})]) 
    \end{bmatrix} \tag{10}

The main thing to notice is that it is triangular, which means the determinant
is just the product of the diagonals.  The first :math:`x_{1:d}` variables are
unchanged, so those entries in the Jacobian are just the identify function and
zeros, while the other :math:`x_{d+1:D}` vars are scaled by the :math:`exp(s(\cdot))`
values (so it's gradient is just the value it is scaled by).  The other
non-zero, non-diagonal part of the Jacobian can be ignored because it's never
used.  Putting this all together, the logarithm of the Jacobian determinant
simplifies to:

.. math::

    \log\Big(\big|det\big(\frac{\partial y}{\partial x^T}\big)\big| \Big) = 
    \sum_j s(x_{1:d})_j
    \tag{11}

which is just the sum of the scaling values (all the other diagonal values are
:math:`\log (1) = 0`).

.. figure:: /images/realnvp_masks.png
  :height: 270px
  :alt: Masking Scheme for Coupling Layers
  :align: center

  **Figure 3: Masking schemes for coupling layers indicated by black and white:
  spatial checkboard (left) and channel wise (right).  Squeeze operation (right) indicated by numbers. [1]**

Partitioning the variables is an important choice since you will want to make
sure you have good "mixing" of dimensions.  [1] proposes two schemes where
:math:`d=\frac{D}{2}`.  Figure 3 shows these two schemes with black and white
squares.  Spatial checkboarding masking simply uses an alternating pattern to
partition the variables, while channel-wise partitions the channels.

Although it may seem tedious to code up Equation 8, one can simply implement the
partitioning schemes by providing a binary mask :math:`b` (as shown in Figure 3) and use
an element-wise product:

.. math::

   y = b \odot x + (1-b) \odot (x \odot exp(s(b\odot x))  + t(\odot x)) \tag{12}

Finally, the choice of architecture for :math:`s(\cdot)` and :math:`t(\cdot)`
functions is important.  The paper uses ResNet blocks as a backbone to define
these functions with additional normalization layers (see more details on these
and other modifications I did below).  But they do use few interesting things
here:

1. On the output of the :math:`s` function, they use a `tanh` activation
   multiplied by a learned scale parameter.  This is presumably to mitigate the
   effect of using `exp(s)` to scale the variables.  Directly using the outputs
   of a neural network could cause big swings in :math:`s` leading to blowing up
   :math:`exp(s)`.
2. To this point, they also add a small :math:`L_2` regularization on :math:`s`
   parameters of :math:`5\cdot 10^{-5}`.
3. On the output of the :math:`t` function, they just use an affine output
   since you want :math:`t` to be able to shift positive or negative.

Stacking Coupling Layers
------------------------

As mentioned before, coupling layers are only useful if we can stack them,
otherwise half of the variables would be unchanged.  By using alternating
patterns of spatial checkboarding and channel wise masking with multiple
coupling layers, we can ensure that the deep net touches every input variable
and that it has enough capacity to learn the necessary invertible transform.
This is directly analgous to adding layers in a feed forward network (albeit
with more complexity in the loss function).

The Jacobian determinant is straightforward to compute using the multi-variate
product rule:

.. math::

    \frac{\partial f_b \circ f_a}{\partial x_a^T}(x_a) &= 
    \frac{\partial f_a}{x_a^T}(x_a) \cdot \frac{\partial f_b}{x_b^T}(x_b = f_a(x_a)) \\
    det(A\cdot B) &= det(A)det(B) \\
    \log\big(\big|det(A\cdot B)\big|\big) &= \log det(A) + \log det(B) && \text{since all scaling factors are positive} \\
    \tag{13}

So in our loss function, we can simply add up all the Jacobian determinants of
our stacked layers to compute that term.

Similarly, the inverse can be easily computed:

.. math::

   (f_b \circ f_a)^{-1} = f_a^{-1} \circ f_b^{-1} \tag{14}

which basically is just computing the inverse of each layer in reverse order.

.. admonition:: Data Preprocessing and Density Computation

    A direct consequence of Equation 5-7 is that *any* pre-processing
    transformations done to the training data needs to be accounted for
    in the Jacobian determinant.  As is standard in neural networks,
    the input data is often pre-processed to a range usually in some interval
    near :math:`[-1, 1]` (e.g. shifting and scaling normalization).
    If you don't account for this in the loss function, you are not actually
    generating a probability and the typical comparisons you see in papers
    (e.g. bits/pixel) are not valid.  For a given pre-processing function
    :math:`x_{pre} = h(x)`, we can update Equation 6 as such:

    .. math::
    
        \log p_X(x) &= \log p_Z(f_\theta(h(x))) + \log\Big(\big|det\big(\frac{\partial f_\theta(h(x))}{\partial x}\big)\big| \Big)\\
        &= \log p_Z(f_\theta(h(x))) + \log\Big(\big|det\big(\frac{\partial f_\theta(x_{pre} = h(x))}{\partial x_{pre}}\big)\big| \Big) 
            + \log\Big(\big|det\big(\frac{\partial h(x)}{\partial x}\big)\big|\big) \\
        \tag{15}

    This is just another instance of "stacking" a pre-processing step (i.e.
    function composition).

    For images in particular, many datasets will scale the pixel values
    to be between :math:`[0, 1]` from the original domain of :math:`[0, 255]`
    (or :math:`[0, 256]` with uniform noise; see 
    `my previous post <link://a-note-on-using-log-likelihood-for-generative-models>`__).
    This translates to a per-pixel scaling of :math:`h(x) = \frac{x}{255}`.  Since each
    pixel is independently scaled, this corresponds to a diagonal Jacobian determinant:
    :math:`\frac{1}{255} I` where :math:`I` is the identify matrix, resulting in a simple
    modification to the loss function.

    If you have a more complex pre-processing transform, you will have to do a
    bit more math and compute the respective gradient.  My implementation of
    Real NVP (see below for why I changed it from what's stated in the paper)
    uses a transform of :math:`h(x) = logit(\frac{0.9x}{256} + 0.05)`, which is
    still done independently per dimension but is more complicated than simple scaling.
    In this case, the per pixel derivative is: 
    
    .. math::

        \frac{dh(x)}{dx} = \frac{0.9}{256}\big(\frac{1}{\frac{0.9x}{256} + 0.05} + \frac{1}{1 - (\frac{0.9x}{256} + 0.05)}\big) \tag{16}

    It's not the prettiest function but also simple enough to compute since you
    still have a diagonal Jacobian.

Multi-Scale Architecture
------------------------

With the above concepts, Real NVP uses a multi-scale architecture to reduce
the computation burden and distributing the loss function throughout the
network.  There are two main ideas here: (a) a squeeze operation to transform
a tensor's spatial dimensions into channel dimensions, and (b) a factoring out
half the variables at regular intervals.

The squeeze operation takes the input tensor and, for each channel, divides it 
into :math:`2 \times 2 \times c` subsquares, then reshapes them into 
:math:`1 \times 1 \times 4c` subsquares.  This effectively reshapes a 
:math:`s \times s \times c` tensor into a :math:`\frac{s}{2} \times \frac{s}{2}
\times 4c` tensor moving spatial size to the channel dimension.
Figure 3 shows the squeeze operation (look at how the numbers are mapped on the
left and right sides).

The squeeze operation is combined with coupling layers to define the basic
block of the Real NVP architecture with consists of: 

* 3 coupling layers with alternative checkboard masks
* Squeeze operation
* 3 more coupling layers with alternating channel-wise mask 

Channel-wise masking makes more sense with more channels so having it follow
the squeeze operation is sensible.  Additionally, since half of the variables
are passed through, we want to make sure there is no redundancy from the 
checkboard masking.  At the final scale, four coupling layers are used with
alternating checkboard masking.

At each of the different scales, half of the variables are factored out and 
passed directly to the output of the entire network.  This is done to reduce
the memory and computational cost.  Defining the above
coupling-squeeze-coupling block as :math:`f^{(i)}` with latent variables
:math:`z` (the output of the network), we can recursively define this by:

.. math::

    h^{(0)} &= x \\
    (z^{(i+1)}, h^{(i+1)}) &= f^{(i+1)}(h^{(i)}) \\
    z^{(L)} &= f^{(L)}(h^{(L-1)}) \\
    z &= (z^{(1)}, \ldots, z^{(L)}) \tag{17}

where :math:`L` is the number of coupling-squeeze-coupling blocks.
At each iteration, the spatial resolution is reduced and the 
number of hidden layer channels in the :math:`s` and :math:`t` ResNet is
doubled.  

The factored out variables are concatenated out to generate the final latent
variable output.  This factoring helps propagate the gradient more easily
throughout the network instead of having it go through many layers. 
The result is that each scale learns different levels of layers of features
from local, fine-grained to global, coarse ones.  I didn't do any experiments
on this aspect but you can see some examples they did in Appendix D of [1].

A final note in this subsection that wasn't obvious to me the first time I read
the paper: the number of latent variables you use is *equal* to the input
dimension of :math:`x`!  While models like VAEs or GANs usually have a much
smaller latent representation, we're using many more variables.  This makes
perfect sense because our network is invertible so you need the same number
of input and output dimensions but it seems inefficient!  This is another
reason why I'm skeptical of the representation power of these stacked coupling
layers.  The problem may be "easier" because you have so many latent variables
where you don't really need much compression.  But this is just a random
speculation on my side without much evidence.

Modified Batch Normalization
----------------------------

Experiments
===========


Implementation Notes
--------------------

* ResNet basic block
* Use a convnet to project to my desired hidden layer depth and another one to project back to original depth
* Instance norm
* Use PyTorch multi-scaling
* Make sure you mask out the :math:`s` vars when computing the loss function too!

Conclusion
==========

Further Reading
===============

* Previous posts: `A Note on Using Log-Likelihood for Generative Models <link://a-note-on-using-log-likelihood-for-generative-models>`__
* Wikipedia: `Latent Variable Model <https://en.wikipedia.org/wiki/Latent_variable_model>`__, `Probabilify Density Function <https://en.wikipedia.org/wiki/Probability_density_function#Vector_to_vector>`__, `Inverse Transform Sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`__, `Probability Integral Transform <https://en.wikipedia.org/wiki/Probability_integral_transform>`__, `Change of Variables in the Probability Density Function <https://en.wikipedia.org/wiki/Probability_density_function#Densities_associated_with_multiple_variables>`__
* [1] Dinh, Sohl-Dickstein, Bengio, Density Estimation using Real NVP, `arXiv:1605.08803 <https://arxiv.org/abs/1605.08803>`__, 2016
* [2] Stanforrd CS236 Class Notes, `<https://deepgenerativemodels.github.io/notes/flow/>`__

.. [1] Apparently, autoregressive models can be interpreted as flow-based models (see [2]) but it's not very intuitive to me so I like to think of them as their own separate thing.
