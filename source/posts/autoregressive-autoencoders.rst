.. title: Autoregressive Autoencoders
.. slug: autoregressive-autoencoders
.. date: 2017-10-05 08:14:15 UTC-04:00
.. tags: autoencoders, autoregressive, generative models, MADE, MNIST, mathjax
.. category: 
.. link: 
.. description: A writeup on Masked Autoencoder for Distrbution Estimation (MADE).
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


You might think that I'd be bored with autoencoders by now but I still
find them extremely interesting!  In this post, I'm going to be explaining
a cute little idea that I came across in the paper `MADE: Masked Autoencoder
for Distribution Estimation <https://arxiv.org/pdf/1502.03509.pdf>`_.
Traditional autoencoders are great because they can perform unsupervised
learning by mapping an input to a latent representation.  However, one
drawback is that they don't have a solid probabilistic basis (
(of course there are other variants of autoencoders that do, see previous posts
`here <link://slug/variational-autoencoders>`__, 
`here <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, and
`here <link://semi-supervised-learning-with-variational-autoencoders>`__). 
By using what the authors define as the *autoregressive property*, we can
transform the traditional autoencoder approach into a fully probabilistic model
with very little modification! As usual, I'll provide some intuition, math and
show you my implementation.  I really can't seem to get enough of autoencoders!

.. TEASER_END

|h2| Vanilla Autoencoders |h2e|

The basic `autoencoder <https://en.wikipedia.org/wiki/Autoencoder>`_
is a pretty simple idea.  Our primary goal is take an input sample
:math:`x` and transform it to some latent dimension :math:`z` (*encoder*),
which hopefully is a good representation of the original data.  However, as
usual, what makes a good representation?  A vanilla autoencoder answer: "*A
good representation is one where you can reconstruct the original input!*".
This reconstruction from the latent dimension :math:`z` to the original
input sample :math:`\hat{x}` is called the *decoder*.  Figure 1 shows a picture
of what this looks like.

.. figure:: /images/autoencoder_structure.png
  :width: 400px
  :alt: Vanilla Autoencoder
  :align: center

  Figure 1: Vanilla Autoencoder (source: `Wikipedia <https://en.wikipedia.org/wiki/Autoencoder>`_)

From Figure 1, we typically will use a neural network as the encoder and
a different (usually similar) neural network as the decoder.  Additionally,
we'll typically put a sensible loss function on the output to ensure :math:`x`
and :math:`\hat{x}` are as close as possible:

.. math::

    \mathcal{L_{\text{binary}}}({\bf x}) &= \sum_{i=1}^D -x_i\log \hat{x}_i - (1-x_i)\log(1-\hat{x_i}) \tag{1} \\
    \mathcal{L_{\text{real}}}({\bf x}) &= \sum_{i=1}^D  (x_i - \hat{x}_i)^2 \tag{2}

Here we assume that our data point :math:`{\bf x}` has :math:`D` dimensions.
The loss function we use will depend on the form of the data.  For binary data,
we'll use the cross entropy and for real-valued data we'll use the mean squared
error.  These correspond to modelling :math:`x` as a Bernoulli and Gaussian
respectively (see the box).

.. admonition:: Negative Log-Likelihoods (NLL) and Loss Functions

    The loss functions we typically use in training models are usually derived
    by some assumption on the probability distribution of the data.  It just
    doesn't look that way because we typically use the negative log-likelihood
    as the loss function.  We can do this because we're usually just looking
    for a point estimate (i.e. optimizing) so we don't need to worry about the
    entire distribution, just a single point that gives us the highest
    probability.

    For example, if our data is binary, then we can model it as a 
    `Bernoulli <https://en.wikipedia.org/wiki/Bernoulli_distribution>`__ 
    with parameter :math:`p` on the interval :math:`(0,1)`.  The probability
    of seeing a given 0/1 :math:`x` value is then:

    .. math::

        P(x) = p^x(1-p)^(1-x)  \tag{3}
    
    If we take the logarithm and negate it, we get the binary cross entropy
    loss function:

    .. math::

        \mathcal{L_{\text{binary}}}(x) = -x\log p - (1-x)\log(1-p) \tag{4}

    This is precisely the expression from Equation 1, except we replace
    :math:`x=x_i` and :math:`p=\hat{x_i}`, where the former is the observed
    data and latter is the estimate of the parameters that our model gives. 

    Similarly, we can do the same trick with a 
    `normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`__.
    Given a observation of a real value :math:`x`, the probability density
    is given by:

    .. math::
        p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \tag{5}

    Taking the negative logarithm of this function:

    .. math::

        -\log p(x) = 
        \frac{1}{2}\log(2\pi \sigma^2) + \frac{1}{2\sigma^2} (x-\mu)^2 \tag{6}

    Now if we assume that the variance is the same fixed value for all our data
    points, then the only parameter we're optimizing for is :math:`\mu`. So 
    adding and multiplying by a bunch of constants to our main expression
    doesn't really change much so we can just simplify it (when optimizing)
    and still get the same point solution.

    .. math::

        -\log p(x) \approx \mathcal{L_{\text{real}}}(x) = (x-\mu)^2
        \\ \tag{6}

    Here our observation is :math:`x` and our model would produce the
    estimate :math:`\mu` i.e. :math:`\hat{x}`.  I have some more details
    on this in one of my previous posts on 
    `regularization <link://slug/probabilistic-interpretation-of-regularization>`__.
   
|h3| Losing Your Identity |h3e|

Now this is all well and good but an astute observer will notice that unless we
put some additional constraints, we can just set :math:`z=x` i.e. the identity
function.  What better representation for reconstruction than exactly the
original data?  This is generally solved by making it difficult to learn just
the identity function.

The easiest method it to just make the dimensions of :math:`z` smaller than
:math:`x`.  For example, if your image has 900 pixels (30 x 30) then make the
dimensions of :math:`z`, say 100.  In this way, you're "forcing" the
autoencoder to learn a more compact representation.

Another method used in *denoising autoencoders* is to artificially introduce
noise on the input :math:`x' = \text{noise}(x)` (e.g. Gaussian noise) but still
compare the output of the decoder with the clean value of :math:`x`.  The
intuition here is that a good representation is robust to any noise that you
might give it.  Again, this prevents the autoencoder from just learning the
identify mapping (because your input is not the same as your output anymore).

In both cases, you eventually end up with a pretty good latent representation
of :math:`x` that can be used in all sorts of applications such as 
`semi-supervised learning <link://slug/semi-supervised-learning-with-variational-autoencoders>`__.

|h3| Proper Probability Distributions |h3e|

Although vanilla autoencoders have done pretty well in learning a latent
representation in an unsupervised manner, they don't have a proper
probabilistic interpretation.  We put a loss function on the outputs of the
autoencoder in Equation 1/2 that has a probabilistic interpretation but
that doesn't mean our autoencoder will generate a proper distribution of the
data!  Let me explain.

Ideally, we would like the unsupervised autoencoder to learn the distribution
of the data.  That is, if we have our different :math:`x` values, for each one
we would be able to generate a probability (or density) :math:`P(x)`.
Usually this means that as a result we also have a 
`generative models <https://en.wikipedia.org/wiki/Generative_model>`__
where we can do nice things like sample from it (e.g. generate new images).

Implicitly this means that if we have :math:`x_1, \ldots, x_n`, then
each one will be assigned a different value :math:`P(x_1),\ldots,P(x_n)`.
And if we sum over all *possible* :math:`x` values, we should get :math:`1`,
i.e. :math:`\sum_x P(x) = 1`.  For autoencoders, we can show that this property
is not guaranteed.  

Consider two samples :math:`x_1`, and :math:`x_2`.  Let's say (regardless of
what type of autoencoder we use), our neural network "memorizes" these two
samples and is able to reconstruct them perfectly.  That is, pass :math:`x_1`
in and get *exactly* :math:`x_1`; pass :math:`x_2` in and get *exactly*
:math:`x_2`.  This implies the loss from Equation 1/2 in both cases is
:math:`0`.  If we translate take the exponential to translate it to a
probability this means both :math:`P(x_1)=1` and :math:`P(x_0)=1`, which of
course is not a valid probability distribution.

For vanilla autoencoders, we started with some neural network and then try to
apply some sort of probabilistic interpretation and it doesn't quite work.  I
like it when we start the way around: start with a probabilistic model and then
figure out how to use neural networks to help you add more capacity and scale
it.


|h2| Autoregressive Autoencoders |h2e|

So vanilla autoencoders don't quite get us to a proper probability distribution
of the data, but is there a way to modify them to get us there?  Let's review
the `product rule <https://en.wikipedia.org/wiki/Chain_rule_(probability)>`__:

.. math::

    p({\bf x}) = \prod_{i=1}^{D} p(x_i | {\bf x}_{<D})

where :math:`{\bf x}_{<D} = [x_1, \ldots, x_{i-1}]`.  Basically, component
:math:`i` of :math:`{\bf x}` only depends on the dimensions of :math:`j < i`.

So how does this help us? In vanilla autoencoders, each component :math:`x_i`
could depend on any of its components :math:`x_1,\ldots,x_n`, this resulted an
improper probability distribution.  If we start with the product rule, which
guarantees a proper distribution, we can work backwards to map the autoencoder
to this model.

For example, let's consider binary data (say a binarized image).  :math:`x_1`
does not depend on any other components of :math:`{\bf x}`, therefore our
implementation should just need to estimate a single parameter :math:`p_1` for
this pixel.  
How about :math:`x_2` though?  Now we let :math:`x_2` depend *only* on
:math:`x_1` since we have :math:`p(x_2|x_1)`.  How about we use a non-linear
function -- say maybe a neural network -- to learn this mapping?  So we'll have
some neural net that maps :math:`x_1` to the :math:`x_2` output.  Now consider
the general case of :math:`x_j`,  we can have a neural net that maps 
:math:`\bf x_{<j}` to the :math:`x_j` output.  Lastly, there's no reason that
each step needs to be a separate neural network, we can just put it all
together so long as we follow a couple of rules:

1. Each output of the network :math:`\hat{x}_i` represents the probability
   distribution :math:`p(x_i|{\bf x_{<i}})`.
2. Each output can only have connections (recursively) to smaller indexed
   inputs :math:`\bf x_{<i}` and not any of the other ones.

Said another way, our neural net first learns :math:`p(x_1)` (just a single
parameter value in the case of binarized data), then iteratively learns the
function mapping from :math:`{\bf x_{<j}}` to :math:`x_j`.  In this view of the
autoencoder, we are sequentially predicting (i.e. regressing) each dimension of
the data using its previous values, hence this property is called the
*autoregressive property* of autoencoders.

Now that we have a fully probabilistic model that can also be implemented as an
autoencoder, let's figure out how to implement it!



|h3| Masks and the Autoregressive Network Structure |h3e|

* Show picture from paper on autoregressive property

|h3| Generating New Samples |h3e|



|h2| MADE Implementation |h2e|

.. figure:: /images/mnist-made.png
  :width: 400px
  :alt: Generated MNIST images using Autoregressive Autoencoder
  :align: center

  Figure X: Generated MNIST images using Autoregressive Autoencoder


|h3| Implementation Notes |h3e|

* Didn't

# Notes
# - Adding a direct (auto-regressive) connection between input/output seemed to make a huge difference (150 vs. < 100 loss)
# - Actually may have been a bug that caused it?
# - Got to be careful when coding up layers since getting indexes for selection exactly right is important
# - Random order didn't really generate any images that were recognizable

# - Need to set_learning_phase for dropout

* Doing custom layers in Keras is so much nicer than using lower level tensorflow don't you think?





|h2| Conclusion |h2e|


|h2| Further Reading |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, and `Semi-supervised Learning with Variational Autoencoders <link://semi-supervised-learning-with-variational-autoencoders>`__
* "MADE: Masked Autoencoder for Distribution Estimation", Germain, Gregor, Murray, Larochelle, `ICML 2015 <https://arxiv.org/pdf/1502.03509.pdf>`_
* Wikipedia: `Autoencoder <https://en.wikipedia.org/wiki/Autoencoder>`_
* Github code for "MADE: Masked Autoencoder for Distribution Estimation", https://github.com/mgermain/MADE

