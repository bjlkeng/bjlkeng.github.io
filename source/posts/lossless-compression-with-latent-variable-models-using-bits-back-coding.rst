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

Note: Here we are distinguishing between data points (denoted by a bold :math:`\bf x`),
and the different components of that data point (not bolded :math:`x_i`).
Usually we will want to encode *multiple* data points (e.g. images), each of which contains
multiple components (e.g. pixels).

Compression is pretty straight forward by just calling :math:`\text{CODE}`
with the input datum and its respective probability distribution to get
your compressed stream of bits :math:`B`. To decode, simply call the
corresponding :math:`{\bf x}=\text{DECODE}(B, P_x)` function to recover your desired
bits.  Notice that our compressed bit stream needs to be paired with
:math:`P_x` else we won't be able to decode it.

"Model" here can be really simple, we could just do a simple histogram of each
symbol in our data stream, and this could serve as our probability
distribution.  You would expect that this simple approach would not do very well
for complex data, for example, trying to compress images.  A histogram
for each pixels (i.e. symbols) across all the images would be very dispersed,
have relatively small probabilities for any given pixel, and thus have poor
compression (recall the higher the probability of a symbol the better entropy
compression methods work).  

So the intuition here is that we want a model that can accurately predict (via
a probability) our datum and its corresponding symbols to allow the entropy
encoder to efficiently compress.  However, using more complicated models with
entropy encoding that treat each datum differently require a bit more thought
on how to apply them.  Let's take a look at a couple of examples.

|h3| Autoregressive Models |h3e|

An autoregressive models simply uses the chain rule to model the data:

.. math::

    P({\bf x}) = \prod_{i=1}^{D} P_x(x_i | {\bf x}_{<i})  \tag{1}

Each indexed component of :math:`\bf x` (i.e. pixel) is dependent only on the
previous ones, using whatever indexing makes sense.  See my post on `PixelCNN <link://slug/pixelcnn>`__
for more details.

When using this type of model with an ANS entropy encoder, we have to be a bit
careful because it decompresses symbols in the last-in-first-out (stack) order.
Let's take a closer look in Figure 1.


.. figure:: /images/bbans_autoregressive.png
  :width: 700px
  :alt: Entropy Encoding and Decoding with an Autoregressive Model
  :align: center

  Figure 1: Entropy Encoding and Decoding with an Autoregressive Model

First looking at encoding from the left side of Figure 1, notice that we have to
reverse the order of the input data to the ANS encoder (the autoregressive
model receives the input in the usual order).  This is needed because we need
to decode the data in the ascending order for the autoregressive conditions
(see decoding below).  Next, notice that our ANS encoder requires both the
(reversed) input data and the appropriate distributions for each symbol (i.e.
each :math:`x_j` component).  Finally, the compressed data is output, which
(hopefully) is shorter than the original input.

Decoding is shown on the right hand side of Figure 1.  It's a bit more
complicated because we must *iteratively* generate the distributions for each
symbol.  Initially, we'll just decode :math:`x_1` since our model can
unconditionally generate its distribution.  This is the reason why we needed
to reverse our input during encoding.  Then, we generate :math:`x_2|x_1` and so
on for each :math:`x_i|x_{1,\ldots,i-1}` until we've recovered the original
data.  Notice that this is quite inefficient since we have to call the model
:math:`n` times for each component of :math:`\bf x`.

I haven't tried this but it seems like something pretty reasonable to do
(assuming you have a good model and I haven't made a serious logical error).
The only problem with autoregressive models is that they are slow!  Perhaps
that's why no one is interested in this?  Anyways, the next method overcomes
this slowness problem.

|h3| Latent Variable Models |h3e|

Latent variable models have a set of unobserved variables :math:`\bf z` in
addition to the observed ones :math:`\bf x`, giving us a likelihood function
of :math:`P(\bf x|\bf z)`.  We'll usually have a prior distribution for :math:`\bf z`
(implicitly or explicitly), and depending on the model, we may or may not have
access to a posterior distribution (more likely an estimate of it) as well: 
:math:`q(\bf z| \bf x)`.

The major difference with latent variable model is that we need to encode the
latent variables (or else we won't be able to generate the required distributions
for :math:`\bf x`).  Let's take a look at Figure 2 to see how the encoding works.

.. figure:: /images/bbans_latent_encode.png
  :width: 600px
  :alt: Entropy Encoding with a Latent Variable Model
  :align: center

  Figure 2: Entropy Encoding with a Latent Variable Model

Starting from the input data, we need to first generate some value for our
latent variable :math:`\bf z` so that we can use it for our model :math:`P(\bf x|\bf z)`.
This can be obtained either by sampling the prior (or posterior if available),
or really any other method that would generate an accurate distribution for :math:`\bf x`.
Once we have :math:`\bf z`, we we can encode the input data as usual.  The one
big difference is that we also have to encode our latent variable where we can
use the prior distribution.  Notice that we cannot use the posterior here because
we won't have access to :math:`\bf x` at decompression time, therefore, would
not be able to decompress :math:`\bf z`.


.. figure:: /images/bbans_latent_decode.png
  :width: 600px
  :alt: Entropy Decoding with a Latent Variable Model
  :align: center

  Figure 2: Entropy Decoding with a Latent Variable Model

Decoding is shown in Figure 2 and works basically as the reverse of encoding.
The major thing to notice is that we have to do operations in a
last-in-first-out order.  That is, first decode :math:`\bf z`, use it to
generate distributional outputs for the components of :math:`\bf x`, then
use those outputs to decode the compressed to recover our original message.

This is all relatively straight forward if you took time to think about it.
There are some other issues as well around discretizing :math:`z` if it's
continuous but we'll cover that below.  The more interesting question is can we
do better?  The answer is a resounding "Yes!", and that's what this post is all
about.  By using a very clever trick you can get some "bits back" to improve
your compression performance.  Read on to find out more!

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
