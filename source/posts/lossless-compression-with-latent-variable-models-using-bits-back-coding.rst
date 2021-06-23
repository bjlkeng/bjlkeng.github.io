.. title: Lossless Compression with Latent Variable Models using Bits-Back Coding
.. slug: lossless-compression-with-latent-variable-models-using-bits-back-coding
.. date: 2021-01-24 11:34:54 UTC-05:00
.. tags: bits back, lossless, compression, asymmetric numeral systems, variational autoencoder, MNIST, mathjax
.. category: 
.. link: 
.. description: A post on using bits-back coding with latent variable models to do lossless compression.
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

A lot of modern machine learning is somehow about "compression", or maybe to use
a fancier term "representations".  Taking a huge dimensional image space
(e.g. 256 x 256 x 3 = 196608) and somehow compressing it into a 1000 or so
dimensional representation seems like pretty good compression to me!  Unfortunately,
it's not a lossless compression (or representation).  Somehow there must
be a way to use what is learned in that powerful lossy representation to help
us better perform *lossless* compression, right?  Of course there is! (It would
be too anti-climatic of a setup otherwise.)

This post is going to introduce a method to perform lossless compression that
leverages the learned "compression" of a machine learning latent variable
model using the Bits-Back Coding algorithm.  Depending on how you first think
about it, this *seems* like it should either be really easy or not possible at
all.  The reality is kind of in between with an elegant theoretical algorithm
that is brought down by the realities of discretization and imperfect learning
by the model.  In today's post, I'll skim over some preliminaries (mostly
referring you to previous posts), go over the main Bits Back Coding in detail,
and discuss some of the implementation details that I came across when trying
to write a toy version.

.. TEASER_END

|h2| Background |h2e|

A **latent (or hidden) variable model** is statistical model where
you *don't* observe some of the variables.  This creates a relationship graph
(usually a DAG) between observed (either input or output, if applicable) and
latent variables.  Often times the modeller will have an explicit intuition
about the meaning of these latent variables.  Other times (especially for deep
learning models), the modeller doesn't explicitly give these latent variables
meaning, and instead they are learned to represent whatever the optimization
procedure deems necessary.  See the background of my `Expectation Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__ post for more details.

A **variational autoencoder** is a special type of latent variable model that
contains two parts: 

    1. A generative model (aka "decoder") that defines a mapping from some
       latent variables (usually independent standard Gaussians) to your data
       distribution (e.g.  an image).
    2. An approximate posterior network (aka "encoder") that maps from your
       data distribution to your latent variable space.

There's also a bunch of math, a reparameterization trick and some deep nets
strung together to make it all work out relatively nicely.  See my post
of `Variational Autoencoders <link://slug/variational-autoencoders>`__ for more
details.

A special type of lossless compression algorithm called *entropy encoders* exploit
the statistical properties of your input data to compress data efficiently. A
relatively new algorithm to perform this lossless compression is called
**Asymmetrical Numeral Systems** (ANS), which essentially map any input data string 
to a (really large) natural number in a smart way such that frequent (or
predictable) characters/strings get mapped to smaller numbers, while infrequent
ones get mapped to larger ones.  One key feature of this algorithm is that
the compressed data stream is generated in first-in-first-out order (i.e. in a
stack).  This means when decompressing, you want to read the compressed data
stream from back-to-front (this will be useful for our discussion below).
My post on `Lossless Compression with Asymmetric Numeral Systems <link://slug/lossless-compression-with-asymmetric-numeral-systems>`__ gives a much better intuition on the whole process
and discusses a lot of the variations and implementation details.


|h2| Lossless Compression with Probabilistic Models |h2e|

Suppose we wanted to perform lossless compression on datum
:math:`{\bf x}=[x_1,\ldots,x_n]` composed of a vector (or tensor) of symbols
(i.e. an image :math:`\bf x` made up of :math:`n` pixels), and we are given:

1. A model that can tell us the probability of each symbol,
   :math:`P_x(x_i|\ldots)=p_i` (that may or may not have some conditioning on it).
2. A entropy encoder that can return a compressed stream of bits
   given input symbols and their probability distributions call it: 
   :math:`B=\text{CODE}({\bf x}, P_x)`.

Compression is pretty straight forward by just calling :math:`\text{CODE}`
with the input datum and its respective probability distribution to get
your compressed stream of bits :math:`B`. To decode, simply call the
corresponding :math:`{\bf x}=\text{DECODE}(B, P_x)` function to recover your desired
bits.  Notice that our compressed bit stream needs to be paired with
:math:`P_x` else we won't be able to decode it.

"Model" here can be really simple, we could just do a simple histogram of each
symbol in our data stream, and this could serve as our probability
distribution.  You would expect that this simple approach would not do very well
for complex data, for example, trying to compress 1M images.  A histogram
for each pixels (i.e. symbols) across all the images would be very dispersed,
have relatively small probabilities, and thus have poor compression (recall
the higher the probability the better entropy compression methods work).

More complicated models though require a bit more thought on how to apply it
due to the various conditioning of models.  Let's take a look at a couple of
examples.

|h3| Autoregressive Models |h3e|

An autoregressive models simply uses the chain rule to model the data:

.. math::

    P({\bf x}) = \prod_{i=1}^{D} P_x(x_i | {\bf x}_{<i})  \tag{1}

Each indexed component of :math:`\bf x` (i.e. pixel) is dependent only on the
previous ones, using whatever indexing makes sense.  See my post on `PixelCNN <link://slug/pixelcnn>`__
for more details.

When using this type of model with the ANS entropy encoder, we have to be a bit
careful because it decompresses symbols in the last-in-first-out (stack) order.
For example, we should compress in reverse order:

.. math::

    \text{CODE}(x_n, P_x(x_n|x_{n-1}\ldots x_1)), \ldots, \text{CODE}(x_1, P_x(x_1)) \tag{2}
 
Notice that when compressing, you have access to the entire :math:`x` vector so
there's no issue there.  When decompressing, you only have access to the
model and any symbols you decoded so far, so you must decode in the appropriate order:

.. math::

    \text{DECODE}(B_1, P_x(x_1)), \ldots, \text{DECODE}(B_n, P_x(x_n|x_{n-1}\ldots x_1)) \tag{3}

where I'm just using a convenience notation for the bitstream :math:`B_i` to
represent the partial bit stream at that point in the decoding.  Overall, it's
pretty straight forward using an autoregressive model so long as you keep in mind
the order of encoding.  

I haven't tried this but it seems like something pretty reasonable to do
(assuming you have a good model and I haven't made a serious logical error).
The only problem with autoregressive models is that they are slow!  During
encoding, it should be quick because you can call the model in parallel by
applying :math:`x` on the input.  But on decoding, you have to iteratively call
the model with each new piece of data decoded (e.g. once for each of
:math:`P_x(x_1)`, :math:`P_x(x_2|x1)`, :math:`P_x(x3|x_2,x_1)`, etc.) Perhaps
that's why no one is interested in this?

|h3| Latent Variable Models |h3e|

Latent variable models have a set of unobserved variables :math:`\bf z` in
addition to the observed ones :math:`\bf x`, giving us a likelihood function
of :math:`P(\bf x|\bf z)`.  We'll usually have a prior distribution for :math:`\bf z`
(implicitly or explicitly), and depending on the model, we may or may not have
access to a posterior distribution (or an estimate of it) as well: :math:`P(\bf z| \bf x)`.

To start, let's think about how we would encode :math:`\bf x` with just the
likelihood.  First, we would need some value of :math:`\bf z` so that we can
get a distribution using :math:`P(\bf x|\bf z)`.  This can be obtained either
by sampling the prior, or using its mean, or any other method you wish.
After that, it's pretty straight forward to encode:

.. math::

    \text{CODE}(x_1, P(x_1|{\bf z}), \ldots, \text{CODE}(x_n, P(x_n|{\bf z})), 
    \text{CODE}({\bf z}, P({\bf z})) \tag{4}

Notice the big difference here is that we need to encode the latent variables
at the end.  We can use the prior distribution to encode the latent variables
using ANS since that the best guess as to how they are distributed.  The main
issue with this method is that it's probably not very good.  Recall, we're
using a generic :math:`\bf z` that's sampled from the prior, it's unlikely that
our sample :math:`\bf x` is going to be probable under that :math:`\bf z`, thus
poor compression.  There are some other issues as well around discretizing
:math:`z` if it's continuous but we'll cover that below.

Things change though if we do have access to a posterior.  Instead of our
generic :math:`\bf z`, we can draw from our posterior distribution (or point
estimate) using :math:`P({\bf z|x})`.  The encoding would be the same as
Equation 4 but most likely with better compression due higher probabilities
from our likelihood.  

This is all relatively straight forward if you took time to think about it but
!he question is can we do better?  Yes!  And that's what this post is all
about.  Using a very clever trick you can get "bits back" which we'll cover
below.

|h2| Bits Back Coding |h2e|


The most straight forward approach for lossless c


* Can't directly use the latent representation
* Instead use it to define probability distribution to use for entropy encoder
* Show how to map
* Make a figure to show progression
* Show some equations

|h2| Implementation Details |h2e|

|h2| Experiments |h2e|

Note that the idea is that the probabilistic model will be widely applicable within a domain
(e.g. trained on ImageNet) so that the compression algorithm will package it so it doesn't need
to be distributed along with the compressed message each time.


|h2| References |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `Lossless Compression with Asymmetric Numeral Systems <link://slug/lossless-compression-with-asymmetric-numeral-systems>`__, `Expectation Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__
* My implementation on Github: `notebooks <https://github.com/bjlkeng/sandbox/tree/master/bitsback>`__
* [1] "Practical Lossless Compression with Latent Variables using Bits Back Coding", Townsend, Bird, Barber, `ICLR 2019 <https://arxiv.org/abs/1901.04866>`__
* [2] "Bit-Swap: Recursive Bits-Back Coding for Lossless Compression with Hierarchical Latent Variables", Kingma, Abbeel, Ho, `ICML 2019 <https://arxiv.org/abs/1905.06845>`__
