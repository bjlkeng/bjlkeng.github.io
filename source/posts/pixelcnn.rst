.. title: PixelCNN
.. slug: pixelcnn
.. date: 2019-07-08 08:11:09 UTC-04:00
.. tags: generative models, autoregressive, CIFAR10, mathjax
.. category: 
.. link: 
.. description: A post of PixelCNN generative models.
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

It's been a long time coming but I'm finally getting this post out!  I read
this paper a couple of years ago and wanted to really understand it because it
was state of the art at the time (still pretty close even now).  As usual
though, once I started down the variational autoencoder line of posts, there
was always *yet* another VAE paper to look into so I never got around to
looking at this one.

This post is all about a proper probabilistic generative model called Pixel
Convolutional Neural Networks or PixelCNN.  It was originally proposed
as a side contribution of Pixel Recurrent Neural Networks in [1] and later
expanded upon in [2,3] (and I'm sure many other papers).  The real cool thing
about it is that it's (a) probabilistic, and (b) autoregressive.  It's still
counter-intuitive to me that you can generate images one pixel at at time, but
I'm jumping ahead of myself here.  We'll go over some background material, the
method, and my painstaking attempts at an implementation (and what I learned
from it).  Let's get started!

.. TEASER_END

|h2| Autoregressive Generative Models |h2e|

Before we begin, we should review autoregressive generative models.
I'll basically summarize what I wrote in one of my previous post: 
`Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__.

An `Autoregressive model <https://en.wikipedia.org/wiki/Autoregressive_model>`__
is usually used in the context of time-series modelling 
(of `random processes <https://en.wikipedia.org/wiki/Stochastic_process>`__)
where :math:`y_n` depends on :math:`y_{n-1}` or some earlier value. 
In particular, literature usually assume a linear dependence and name these
"AR" models of the "ARIMA" notoriety.  Here "auto" refers to self, and
"regressive" means regressed against.  

In the conext of deep generative models, we'll drop the condition of linear
dependence and formulate our image problem as a random process.  In particular,
we will:

a. Use a deep generative models (obviously non-linear), and
b. Assume the pixels of an image is a random variable with a specific ordering
   (top to bottom, left to right), which formulates it as a random process.

With that in mind, let's review the 
`product rule <https://en.wikipedia.org/wiki/Chain_rule_(probability)>`__:

.. math::

    p({\bf x}) = \prod_{i=1}^{D} p(x_i | {\bf x}_{<i})  \tag{8}

where :math:`{\bf x}_{<i} = [x_1, \ldots, x_{i-1}]`.  Basically, component
:math:`i` of :math:`{\bf x}` only depends on the dimensions of :math:`j < i`.
In your head, you can think of each :math:`x_i` as a pixel.  So each pixel is
going to have a probability distribution that is a function of all the
(sub-)pixels that came before it (for RGB images, each of "R", "G", "B" are
treated as separate sub-pixels).

The way to generate an image from an autoregressive generative model is as follows:

1. Naturally, the first (sub-)pixel in our sequence has nothing before it so it's
   a totally unconditional distribution.  We simply sample from this
   distribution to get a concrete realization for the first (sub-)pixel.
2. Each subsequent (sub-)pixel distribution is generated in sequence conditioned on
   all previously sampled (sub-)pixels.  We simply sample from this conditional
   distribution to get the current pixel value.
3. Repeat until you have the entire image.

According to my intuition, this is a really weird way to generate an image!
Think of it, if you want to generate a picture of a dog, you start at the top
left pixel, figure out what it is, then move on to the one beside it, and so
on.  This goes against my sensibilities of implicitly having a hierarchical
relationship from a higher level concept like a dog and low-level pixels.
In any case, we can still get good negative log-likelihood (although the quality
of the images are another story).


|h2| PixelCNN |h2e|

Now that we covered autoregressive generative models, PixelCNN is
not too difficult to understand.  We want to build a *single* CNN that takes as
input an image and outputs a *distribution* for each (sub-)pixel (theoretically,
you could have a different network for each pixel but that seems inefficient).
There are a couple subtleties when doing that:

(a) Due to the autoregressive nature, pixel :math:`i` should not see any pixels
    :math:`\geq i`, otherwise it wouldn't be autoregressive (you could "see the
    future").
(b) Selecting a distribution and its corresponding loss function in order to
    model the output pixels.

Let's take a look at each one separately.

|h3| Masked Convolution |h3e|

The masked convolution is basically the same idea as the masked autoencoder
from my post on MADE: 
`Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__.
As stated above, we impose (an arbitrary) ordering on the sub-pixels: 
top to bottom, left to right, R to G to B.  Given this, we want to make sure
pixels :math:`\geq i` are "hidden" from pixel :math:`i`, we can accomplish
this with *masks*.  This is shown on the left side of Figure 1 where pixel
:math:`i` only "reads" from its predecessors (the center image is the same
thing for a larger multi-scale resolution).  This can easily be generated using
a "mask" that is element-wise multiplied by your convolution kernel.  We'll
talk about the implementation of this later on.

.. figure:: /images/pixelcnn_mask.png
  :width: 600px
  :alt: PixelCNN Mask
  :align: center

  Figure 1: PixelCNN Mask (source: [1])

However, the mask doesn't just deal with the spacial location of full pixels,
it also has to take into account the RGB values, which leads us to two
different types of masks: A and B.  For the first convolution
layer on the original image, the same rule applies as above, never read ahead.
For full pixel :math:`i`'s mask when reading from pixel :math:`x_i`, the "B"
pixel should only be connected to "G" and "R"; the "R" pixel should only be
connected to "R"; and the "R" pixel shouldn't be connected at all to pixel
:math:`i`.  You can see the connectivity on the right side of Figure 1
(Note you still have full "read" access to all sub-pixels from predecessor
pixels, the differences for these masks only affect the current pixel).

For layers other than the first one, things change.  Since any sub-pixel output
from a convolution layer has already been masked from it's corresponding
input sub-pixel, you are free to use it in another convolution.  That might
be a bit confusing but you can see this in Mask B in Figure 1.  For example,
the output of the "G" sub-pixel in Mask B, depends on the "G" sub-pixel output
from Mask A, which in turn depends only on the "R".  That means "G" only depends
on "R", which is what we wanted.  If we didn't do this, the "G" output from
Mask B would never be able to "read" from the "R" sub-pixel in the original
input.  If this is just confusing, just take a minute to study the connectivity
in the diagram and I think it should be pretty clear.

Keep in mind Figure 1 really only shows the case where you have three color
channels.  In all convolution layers beyond the first one, we likely have many
different filters per layer.  Just imagine instead of "RGB", we have
"RRRRGGGGBBBB" as input to Mask B, and you can probably guess what the
connectivity should look like.  Again this only applies to the current
sub-pixels, we should have full connectivity to all predecessors.

|h3| Discretized Logistic Mixture Likelihood |h3e|

I covered this topic in my previous post in detail
(`Importance Sampling and Estimating Marginal Likelihood in Variational Autoencoders <link://slug/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders>`__) so I'll just do a summary here.

Modelling pixels is a funny problem.  On one hand, we can view them as 8-bit
discrete random variables (e.g. 256-way softmax), but on the other hand, we can
give them as approaching a sort-of continuous distribution (e.g. real-valued output).
The former approach is the way that [1] tackles the problem: just stick a
256-softmax output on each sub-pixel.  There are two issues: (1) it uses up a
gigantic amount of resources, and (2) there is no relationship between adjacent
pixel values (R=100 is similar enough to R=101).  Mostly due to the first reason,
I opted out to not implement it (couldn't fit a full image on my meager GTX 1070).

The other way to go about it is to imagine the pixel generation process as such described in [2]:

(1) For each sub-pixel, generate a continuous distribution :math:`\nu`
    representing the intensity.  For example, :math:`\nu` could be a 
    `logistic distributions <https://en.wikipedia.org/wiki/Logistic_distribution>`__
    parameterized by :math:`\mu, s`.  
(2) Next, "round" each sub-pixel to a
    discretized distribution over :math:`[0, 255] \in \mathbb{Z}` by
    integrating over the appropriate width along the real line (assuming a
    logistic distribution):

    .. math::
    
        P(x|\mu,s) = 
            \begin{cases}
                \sigma(\frac{x+0.5-\mu}{s}) & \text{for } x = 0 \\
                \sigma(\frac{x+0.5-\mu}{s}) - \sigma(\frac{x-0.5-\mu}{s}) 
                    & \text{for } 0 < x < 255 \\
                1 - \sigma(\frac{x-0.5-\mu}{s}) & \text{for } x = 255
            \end{cases}
        \tag{1}

    where :math:`\sigma` is the sigmoid function (recall sigmoid is the CDF of
    the logistic distribution).
    Here we basically take the :math:`\pm 0.5` interval around each pixel value
    to compute its discretized probability mass .  For the edges, we just
    integrate to infinity.

However, this will make each pixel uni-modal, which doesn't afford us much
flexibility.  To improve it, we use a mixture of logistics for :math:`\nu`:

.. math::

    \nu \sim \sum_{i=1}^K \pi_i logistic(\mu_i, s_i) \tag{2}

To bring it back to neural networks, for each sub-pixel, we want our network
to output three things: 

* :math:`K` :math:`\mu_i` representing the centers of our logistic distributions
* :math:`K` :math:`s_i` representing the scale of our logistic distributions
* :math:`K` :math:`\pi_i` representing the mixture weights (summing to 1)

We don't need that many mixture components to get a good result ([2] uses
:math:`K=5`), which is a lot less than a 256-way softmax.  Figure 2 shows
a realization of the toy discretized mixture I did while testing.  You can see,
we clearly that we can model multi-modal distributions, and at the same time
have a lot of mass at the 0 and 255 pixels.

.. figure:: /images/pixelcnn_histogram.png
  :width: 400px
  :alt: Distribution of Discretized Logistic Mixtures
  :align: center

  Figure 2: Distribution of Discretized Logistic Mixtures: Top to
  bottom represent the "R", "G", "B" components of a single pixel

All of this can be implemented pretty easily by strapping a bunch of masked
convolution layers together and setting the last layer to have :math:`3*3K`
filters (one for each sub-pixel).  The really interesting stuff happens in the
loss function though which computes the negative log of Equation 1.  The nice thing
about this loss is that it's fully probabilistic from the start.
We'll get to implementing it in a later section but suffice it to say, it's not
easy dealing with overflow!

|h3| Training and Generating Samples |h3e|

Training this network is actually pretty easy, all we have to do is make
the actual image for available on the input and output of the network.  
Note the input dimensions of the network naturally takes an image, the output
of the network outputs a distribution for each sub-pixel.  However, a custom
loss function is needed that takes all the outputs of the network and the
actual image and compute the negative log-likelihood.  Other than the complexity
of the loss function, you just set it and go!

Generating images is something that is a bit more complicated but follows the same idea from my post on `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__:

0. Set :math:`i=0` to represent current iteration and pixel (implicitly we
   would translate it to the row/col/sub-pixel in the image).
1. Start off with a tensor for your image :math:`\bf x^0` with any initialization
   (it doesn't matter).
2. Feed :math:`\bf x^i` into the PixelCNN network to generate distributional
   outputs :math:`\bf y^{i+1}`.
3. Randomly sample sub-pixel :math:`u_{i+1}` from the mixture distribution
   defined by :math:`\bf y^{i+1}` (we only need the subset of values for
   sub-pixel :math:`i+1`).
4. Set :math:`x^{i+1}` as :math:`x^i` but replacing the single sub-pixel with
   :math:`u_{i+1}`.
5. Repeat step 2-4 until entire image is generated.

This is a slow process!  For a 32x32x3 image, we basically need to do a forward
pass of the network 3072 times for a single image (of course we have some
parallelism because we can do several images in batch).  This is the downside
of autoregressive models: training is done in parallel but generation is
sequential (and slow).  As a data point, my slow implementation took almost 37
mins to generate 16 images (forward passes were parallelized on the GPU but
sampling was sequential in a loop on the CPU).

|h2| Implementation Details |h2e|

So far the theory isn't too bad: some masked layers, extra outputs and some
sigmoid losses and we're done, right?  Well it's a bit tricker than that,
especially the loss function.  I'll explain details (and headaches) that I went
through implementing it in Keras.  As usual, you can find all my code in this 
`Github repo <https://github.com/bjlkeng/sandbox/tree/master/notebooks/pixel_cnn>`__.

|h3| Masked Convolution Layer |h3e|

|h3| Loss Function |h3e|

|h3| Implementation Strategy |h3e|


|h2| Experiments |h2e|

|h2| Conclusion |h2e|

|h2| Further Reading |h2e|

* [1] "Pixel Recurrent Neural Networks," Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu, `<https://arxiv.org/abs/1601.06759>`__.
* [2] "PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications," Tim Salimans, Andrej Karpathy, Xi Chen, Diederik P. Kingma, `<http://arxiv.org/abs/1701.05517>`__.
* [3] "Conditional Image Generation with PixelCNN Decoders," Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, Koray Kavukcuoglu, `<https://arxiv.org/abs/1606.0532A>`__
* Wikipedia: `Autoregressive model <https://en.wikipedia.org/wiki/Autoregressive_model>`__
* Previous posts: `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__, `Importance Sampling and Estimating Marginal Likelihood in Variational Autoencoders <link://slug/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders>`__


