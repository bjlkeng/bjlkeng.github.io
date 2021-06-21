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
**Asymmetrical Numeral Systems**, which essentially map any input data string 
to a (really large) natural number in a smart way such that frequent (or
predictable) characters/strings get mapped to smaller numbers, while infrequent
ones get mapped to larger ones.  One key feature of this algorithm is that
the compressed data stream is generated in first-in-first-out order (i.e. in a
stack).  This means when decompressing, you want to read the compressed data
stream from back-to-front (this will be useful for our discussion below).
My post on `Lossless Compression with Asymmetric Numeral Systems <link://slug/lossless-compression-with-asymmetric-numeral-systems>`__ gives a much better intuition on the whole process
and discusses a lot of the variations and implementation details.

|h2| Bits Back Coding |h2e|

* Can't directly use the latent representation
* Instead use it to define probability distribution to use for entropy encoder
* Show how to map
* Make a figure to show progression
* Show some equations

|h2| Implementation Details |h2e|

|h2| Experiments |h2e|


|h2| References |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `Lossless Compression with Asymmetric Numeral Systems <link://slug/lossless-compression-with-asymmetric-numeral-systems>`__, `Expectation Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__
* My implementation on Github: `notebooks <https://github.com/bjlkeng/sandbox/tree/master/bitsback>`__
* [1] "Practical Lossless Compression with Latent Variables using Bits Back Coding", Townsend, Bird, Barber, `ICLR 2019 <https://arxiv.org/abs/1901.04866>`__
* [2] "Bit-Swap: Recursive Bits-Back Coding for Lossless Compression with Hierarchical Latent Variables", Kingma, Abbeel, Ho, `ICML 2019 <https://arxiv.org/abs/1905.06845>`__
