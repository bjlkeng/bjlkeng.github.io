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

  Figure 3: Entropy Decoding with a Latent Variable Model

Decoding is shown in Figure 3 and works basically as the reverse of encoding.
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

From the previous section, we know that we can encode and decode data using a
latent variable model with relative ease.  The big downside is that we're
"wasting" space by encoding the latent variables.  They're necessary to
generate the distributions for our data, but otherwise are not encoding any of
our signal.  It turns out we can use a clever trick to recover some of
this "waste".

Notice in Figure 2, we randomly sample from (an estimate of) the posterior distribution.
In some sense, we're introducing new information from the random sample here
that we must encode.  Instead, why don't we utilize some of the existing bits
we've encoded to get a pseudo-random sample?  Figure 4 shows the encoding
process in more detail.

.. figure:: /images/bbans_bb_encode.png
  :width: 600px
  :alt: Bits Back Encoding with a Latent Variable Model
  :align: center

  Figure 4: Bits Back Encoding with a Latent Variable Model

The key difference here is that we're decoding the existing bitstream (from
previous data that we've compressed) to generate a (pseudo-) random sample :math:`\bf z`
using the posterior distribution.  Since the existing bitstream was encoded using
a different distribution, the sample we decode should *sort of* random.  The nice part
about this trick is that we're still going to encode :math:`\bf z` as usual so any
bits we've popped off the bitstream we get "back" (that is, don't require to be
on the bitstream anymore).  This *reduces* the effective average size of encoding
each datum + latent variables.

.. figure:: /images/bbans_bb_decode.png
  :width: 600px
  :alt: Bits Back Decoding with a Latent Variable Model
  :align: center

  Figure 5: Bits Back Decoding with a Latent Variable Model

Figure 5 shows decoding with Bits Back.  It is the same as latent variable
decoding with the exception that we have to "put back" the bits we took off
originally.  Since our ANS encoding and decoding are lossless, the bits we
put back should be exactly the bits we took off.  The number of bits we remove
will be dependent on the posterior distribution and which bits are in the
stream.

.. figure:: /images/bbans_bitstream_view.png
  :width: 600px
  :alt: Visualization of Bitstream for Bits Back Coding
  :align: center

  Figure 6: Visualization of Bitstream for Bits Back Coding

To get a better sense of how it works, Figure 6 shows a visualization of
encoding and decoding two data points.  Colors represent the different
data: green for existing bitstream, blue for :math:`\bf x^1`, and orange for :math:`\bf x^2` 
(superscript represents data point index).  The different shades represent either
observed data :math:`\bf x` or latent variable :math:`\bf z`.

From Figure 6, the first step in the process is to *reduce* the bitstream length
by (pseudo-)randomly sampling :math:`\bf z`.  This is followed by encoding
:math:`\bf x` and :math:`\bf z` as usual.  Even though we have to encode :math:`\bf z`,
the effective size of the encoding is shorter because of the initial "bits back" we got.
The decoding process the reverse operation of the encoding, including putting
the "bits back" onto the bitstream.

|h3| Theoretical Limit of Bits Back Coding |h3e|

Turning back to some more detailed mathematical analysis, let's see how good
Bits Back is theoretically.  We'll start off with a few assumptions:

1. Our data :math:`\bf x` and latent variables :math:`\bf z` are sampled from
   the true joint distribution :math:`P({\bf x, z})=P({\bf x|z})P({\bf z})`,
   which we have access to.  Of course in the real world, we don't have the
   true distribution, just an approximation.  But if our model is very good, it
   will hopefully be very close to the true distribution.
2. We have access to an approximate posterior :math:`q({\bf z|x})`.
3. Assume we have an entropy coder so that we can optimally code any data point.
4. The pseudo-random sample we get from Bits Back coding is drawn from the approximate posterior :math:`q({\bf z|x})`.

As noted above, if we naively use the latent variable encoding from Figure 2,
given a sample :math:`(x, z)`, our expected message length should be 
:math:`-(\log P({\bf z}) + \log P({\bf x|z}))` bits long.  This uses the fact
(roughly speaking) that the theoretical limit of the number of bits needed to
represent a symbol (in the context of its probability distribution) is its
`information <https://en.wikipedia.org/wiki/Information_content>`__.

However using Bits Back with an approximate posterior :math:`q({\bf z|x})`
for a given *fixed* data point :math:`\bf x`, we can calculate the expected
message length over all possible :math:`\bf z` drawn from :math:`q({\bf z|x})`.
The idea is that we're (pseudo-)randomly drawing :math:`\bf z` values, which
affect each part of the process (bits back, encoding :math:`x`, and encoding
:math:`z`) so we must average (i.e. take the expectation) over it:

.. math::

   L(q) &= E_{q({\bf z|x})}(-\log P({\bf z}) - \log P({\bf x|z}) + \log q({\bf z|x})) \\
        &= \sum_y q({\bf z|x})(-\log P({\bf z}) - \log P({\bf x|z}) + \log q({\bf z|x}))  \\
        &= -\sum_y q({\bf z|x})\log \frac{P({\bf x, z})}{q({\bf z|x})}  \\
        &= -E_{q({\bf z|x})}\big[\log \frac{P({\bf x, z})}{q({\bf z|x})}\big]  \\
        \tag{2}

Equation 2 is also known as the evidence lower bound (ELBO) (see my previous 
`post on VAE <link://slug/semi-supervised-learning-with-variational-autoencoders>`__ 
for more details).  The nice thing about the ELBO is that many ML models (including
the variational autoencoder) use it as its objective function.  So by optimizing our
model, we're simultaneously optimizing our message length.

From Equation 2, we can also see that it is optimized when :math:`q({\bf z|x})` equals
to the true posterior :math:`P({\bf z|x})`:

.. math::

    -E_{q({\bf z|x})}\big[\log \frac{P({\bf x, z})}{q({\bf z|x})}\big]
    &= -E_{q({\bf z|x})}\big[\log \frac{P({\bf z|x})P({\bf x})}{P({\bf z|x})}\big]  && \text{since } 
    P({\bf x, z}) = P({\bf z|x})P({\bf x}) \text{ and } q({\bf z|x})=P({\bf z|x}) \\
    &= -E_{q({\bf z|x})}\big[\log P({\bf x^1})\big] \\
    &= -\log P({\bf x}) \\
    \tag{3}

Which is the optimal code length for sending our data point :math:`x` across.
So *theoretically* if we're able to satisfy all the assumptions then we'll have a
really good encoder!  Of course, we'll never be in this theoretic ideal
situation, we'll discuss some of the issues that reduce it this efficiency.

|h3| Issues Affecting The Efficiency of Bits Back Coding |h3e|

**Transmitting the Model**: All the above discussion assumes that the sender
and receiver have access to the latent variable model but that needs to be sent
as well!  The assumption here is that the model would be so generally applicable
that the compression package would include it by default (or have a plugin) to
download the model.  For example, photos are so common that we conceivably have
a single latent variable model for photos (e.g. something along the lines of
ImageNet).  This would enable the compression encoder/decoder package to include
it and encode images or whatever data distribution it is trained on.  For the
experiments below, I don't include the model size but if I did, it would be
much worse.  I think the benefits are only realized if you can amortize the
cost of sending the model over a huge dataset.

**Discretization**: Another thing that the above discussion glossed over is how
to encoder/decode continuous variables.  ANS (and similar entropy coders) work
on discrete symbols -- not continuous values.  Many popular latent variable
models also have continuous latent variables (e.g. normally distributed), so there
needs to be a discretization step to be able to send them over the wire (we're
assuming the data is discretized already but the same principles would apply).  

Discretization is needed for both the approximate posterior encoding/decoding 
where we (pseudo-)randomly sample our :math:`\bf z` value ("bits back") and for
when we encode/decoder the latent variables using the prior distribution.
Discretization of a sample from the distribution is a relatively simple
operation:

0. Select how many bits you want to use to represent your sample.  This will
   create :math:`2^n` buckets for :math:`n` bits.
1. Partition the distribution's support into :math:`2^n` buckets.  [1] proposed
   equi-probable mass buckets, which is what I implemented.
2. Find the corresponding bucket index (:math:`i`) the point falls in, set the
   discretized value to some value relative to the bucket interval (e.g.
   mid-point of the bucket interval).
3. You can use ANS to encode the discretized value as the symbol :math:`i` with
   an alphabet of :math:`2^n` symbols.  Note: If you use an equi-probable mass
   buckets each symbol will have the same probability, so entropy encoding
   shouldn't do much.

To decode, you essentially can do the reverse operation.  However, there is a
subtlety: you need the sender and receiver to have access to the distribution
in step 1.  The natural choice the prior, which is what is available throughout
the process (assuming you have the model).  You wouldn't be able to use the
posterior because when you are trying to decode :math:`\bf z`, you would need access
to :math:`\bf x`, which you don't have available.  Additionally, you need
to have the same discretization step when you're sampling via "bits back" and
when sending :math:`\bf z` across the wire, or else you lose some precision in
the process.  So the prior is used throughout.

The paper [1] shows that the additional overhead is really just the cost of
discretization.  Since many continuous ML operations work fine with 32-bits
anyways, it shouldn't be a problem.  As the paper suggests, I used a 16-bit
discretization.

**Clean Bits**: The last issue to discuss is the how the bits back operation
samples the :math:`\bf z` value.  Since we're sampling from the bits from the
top of the existing bitstream, which is essentially the previous prior-encoded
posterior sample, we would not expect it to be a true random sample (which
would require a uniform random stream of bits).  Only in the base-case with 
the first sample, can we achieve this either by seeding the bitstream with
truly uniform random bits or by just directly directly sample from the
posterior.  It seems to me like there's not too much to do about this
inefficiency because the whole point of this method is to get "bits back".

|h2| Implementation Details |h2e|

There were three main parts to my toy implementatin: ANS algorithm, variational
autoencoder, and the bitsback algorithm.  Below are some details on each.  You 
can find the code I used here in my `Github <https://github.com/bjlkeng/sandbox/tree/master/bitsback>`__.

**ANS**: The implementation I used was almost identical to the toy implementation I used
from my `previous post on ANS <link://slug/lossless-compression-with-asymmetric-numeral-systems>`__
(which I wrote while travelling down the rabbit hole to understand bitsback algorithm).
That post has some details on the toy implementation that I used for it.  These are
notes for the incremental changes I made:

* I had to fix some slow parts of my implementation or else my experiments
  would have taken forever.  For example, I was calculating the CDF in a slow way using
  native Python data structures.  Switching to numpy `cusum` fixed some of
  that.  Additionally, I had to make sure that all my arrays were in numypy objects
  and not slow native Python lists.
* As part of the algorithm, you have to calculate very big numbers that could
  exceed 64 bits (especially with an alphabet size of 256 and renormalization
  factor of 32).  Python integers are great for this because they have arbitrary size. The
  only thing I had to be careful of was converting between my numpy operations and Python integers.
  It was mostly just wrapping most expressions in the main calculation with `int` but took a bit
  to get it all sorted out.
* Also had to factor the code so that it could code symbols incrementally, instead of having the
  entire message available so it could be used in the bitsback algorithm.
* I used 16 bits of quantization to model the distribution of the 256 pixel values and
  32 bit renormalization.

**Variational Autoencoder**: I just used a vanilla VAE as the latent model since I was just
experimenting on MNIST.  In an effort to modernize, I started with the basic `VAE Keras 2.0 example
<https://raw.githubusercontent.com/keras-team/keras-io/master/examples/generative/vae.py>`__
and added a few modifications:

* I added some ResNet identity blocks to beef up the representation power.  Still not sure
  it really made much of a difference.
* Outputs of the decoder used my implementation of mixture of logistics to model the distributional
  outputs per pixel with 3 components.  I wrote about it a bit in my `PixelCNN
  <http://localhost:8000/posts/pixelcnn/>`__ post.  I'm also wary about whether
  or not this actually made it better.  The original paper [1] just used a Beta-Binomial distribution.
* I used 50 latent dimensions to match [1].

**Bitsback Algorithm**: The bitsback algorithm is conceptually pretty simple but requires you to be a
bit careful in a few areas.  Here are some of the notable point:

* Since the quantization was always using equi-probable bins for a standard normal
  distribution, it made sense to cache the ranges for speeding it up.
* Quantizing the continuous values of the latent variable distributions was
  pain.  For the case of quantizing a standard normal distribution, it was easy
  because, by construction, each bin is equi-probable.  So the distribution is just uniform
  across however many buckets we're using (16-bits in my experiments to match the paper).
* However, if you're trying to quantize a non-standard :math:`\bf z` normal
  distributions using equi-probable bins from a *standard* normal distribution,
  you have to be a bit more careful:

  1. I sampled **2^n** (n=14 in my case) equi-probable values per variable
     from the original :math:`\bf z` distributions using the inverse CDF function from SciPy.
  2. From those sampled values, I made a frequency histogram where the
     buckets correspond to a *standard* normal distribution equi-probable
     buckets.  This represents the quantized distribution, and I coded the ANS algorithm
     to directly use a frequency histogram, which internally is converted to a frequency-CDF.

  I'm not sure if there's a better way to do it but it seemed work well enough.
  You can increase the number of samples to get a more accurate frequency
  distribution but it slows down the algorithm as you might expect.
* Encoding/decoding the :math:`x` pixel values was much easier because they are
  already discretized as 256 pixel values.  It's still a bit slow though since
  I just loop through each pixel value and encode it sequentially. 
* The only tricky part for the pixel values was that I had to translate the probability
  distribution (real numbers) over discrete pixels into a frequency distribution (integers).
  The sum of the frequency distribution also needs to sum to :math:`2^(\text{ANS quant bits})`.
  Additionally, I wanted to ensure that no bin had zero probability, or else if
  you try to encode it, ANS gets super confused.  To pull this off, I just
  multiplied each bin's probability by :math:`2^(\text{ANS quant bits})`, added
  one to each bin, then calculate any excess I have beyond :math:`2^(\text{ANS
  quant bits})` and shave it off the largest frequency bin.  This is obviously
  not an optimal way to do it.  I do wonder if that's why I got results that were worse
  than the paper, but I didn't spend too much time checking.
* Again, I had to be careful not to implicitly convert some of the integer values to floats.
  So in some places, I do some explicit casting of `astype(np.uint64)` so the values don't
  get all mixed up when I send them into ANS.
  
|h2| Experiments |h2e|

My compression results for MNIST (regular, non-binarized) are shown in Table 1.
Very unimpressive if you ask me.  I wasn't really able to get close to the implementation
in [1].  Didn't really try to hard to make it work but I was hoping that I would be
able to at least beat the standard compressors (`bz2` and `gzip`), unfortunately that
didn't happen either.

.. csv-table:: Table 1: Compression Rates for MNIST (bits/pixel)
   :header: "Compressor", "Compression Rates (bits/pixel)"
   :widths: 15, 10
   :align: center

   "My Implementation (Bits Back w/ ANS)", 1.94
   "Bitsback w/ ANS [1]", 1.41
   "Bits-Swap [2]", 1.29
   "bz2", 1.64
   "gzip", 1.42
   "Uncompressed", 8

Interestingly [1] was using a simpler model (a single feed-forward layer for each of the encoder/decoder)
with a Beta-Binomial output distribution for each pixel.  This is obviously is obviously simpler than
my complex ResNet/multi-logistic method.  It's possible that I'm just not able to get a good fit with
my VAE model.  If you take a look in the notebook you'll see that the generated digits I can make
with the decoder look pretty bad.  So this is probably at least part of the reason why I was unable
to achieve good results.

The second reason is that I suspect my quantization isn't so great.  As mentioned above, I did so
funky rounding to ensure no zero-probability buckets, as well as a awkward way to discretize the
latent variables.  I suspect there are some differences from [1]'s implementation 
(which is open source by the way) but I didn't spend too much time trying to
figure out the differences.

In any case, at least my implementation is able to *correctly* encode and decode and somewhat 
approach the proper implementations.  As a toy implementation, I will make the bold assertion
that I coded it in a way that's  a bit more clear than [1]'s implementation so
maybe it's better for educational purposes?  I'll let you be the judge of that.

|h2| Conclusion |h2e|

So there you have it, a method for lossless compression using ML!  This mix of discrete
problems (e.g. compression) and ML is an incredibly interesting direction.  If
I get some time (and who knows when that will be), I'm definitely going to be
looking into some more of these topics.  But on this lossless compression topic, I'm
probably done for now.  There's another topic that I've been excited about recently
and have already started to go down that rabbit hole, so expect one (probably more)
posts on that subject.  Hope everyone is staying safe!

|h2| References |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `Lossless Compression with Asymmetric Numeral Systems <link://slug/lossless-compression-with-asymmetric-numeral-systems>`__, `Expectation Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__
* My implementation on Github: `notebooks <https://github.com/bjlkeng/sandbox/tree/master/bitsback>`__
* [1] "Practical Lossless Compression with Latent Variables using Bits Back Coding", Townsend, Bird, Barber, `ICLR 2019 <https://arxiv.org/abs/1901.04866>`__
* [2] "Bit-Swap: Recursive Bits-Back Coding for Lossless Compression with Hierarchical Latent Variables", Kingma, Abbeel, Ho, `ICML 2019 <https://arxiv.org/abs/1905.06845>`__
