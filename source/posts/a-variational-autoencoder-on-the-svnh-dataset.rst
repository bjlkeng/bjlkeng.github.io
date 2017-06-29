.. title: A Variational Autoencoder on the SVNH dataset
.. slug: a-variational-autoencoder-on-the-svnh-dataset
.. date: 2017-06-27 09:13:03 UTC-04:00
.. tags: variational calculus, svhn, autoencoders, Kullback-Leibler, generative models, mathjax
.. category: 
.. link: 
.. description: A writeup on using VAEs for the SVNH dataset.
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

In this post, I'm going to share some notes on implementing a variational
autoencoder (VAE) on the 
`Street View House Numbers <http://ufldl.stanford.edu/housenumbers/>`_ 
(SVHN) dataset.  My last post on 
`variational autoencoders <link://slug/variational-autoencoders>`__
showed a simple example on the MNIST dataset but because it was so simple I
thought I might have missed some of the subtler points of VAEs -- boy was I
right!  The fact that I'm not really computer vision guy nor a deep learning
guy didn't help either.  Through this exercise, I picked up some of the basics
in the "craft" of computer vision/deep learning area; there are a lot of subtle
points that are easy to gloss over if you're just reading someone else's
tutorial.  I'll share with you some of the details in the math (that I
initially got wrong) and also some of the implementation notes along with a
notebook that I used to train the VAE.  Please check out my previous post
on `variational autoencoders <link://slug/variational-autoencoders>`__,
which I'll assume you're read.

.. TEASER_END

|h2| The Street View House Numbers (SVHN) Dataset |h2e|

The `SVHN <http://ufldl.stanford.edu/housenumbers/>`__ is a real-world image
dataset with over 600,000 digits coming from natural scene images (i.e. Google
Street View images).  It has two formats: format 1 contains the full image with
meta information about the bounding boxes, while format 2 contains just the
cropped digits in 32x32 RGB format.  It has the same idea as the MNIST dataset
but much more difficult because the images are of varying styles, colour and
quality.

.. figure:: /images/svhn_format2.png
  :width: 400px
  :alt: SVHN Format 2 dataset
  :align: center

  **Figure 1: SVNH format 2 cropped digits**

Figure 1 shows a sample of the cropped digits from the SVHN website.  You can
see that you get a huge variety of digits, making the it much harder to train a
model.  What's interesting is that in some of the images, you have several
digits.  For example, "112" in the top row, is centered (and tagged) as "1" but
has additional digits that might cause problems when fitting a model.

|h2| A Quick Recap on VAEs |h2e|

* Reiterate VAE model

|h2| Fixed or Fitted Variance? |h2e|

* Subtlety with math on continuous output variable, need network to encode sigma
* Explain why constant sigma doesn't work (at least not well in this case), the Doersch paper didn't point this part out
* Explain network outputs both mean and var (not the single hyper param in Doersch paper, need to check out Kingma paper, and his corresponding code

* varlog epsilon
  * Point to reddit post

|h2| A Digression on Progress |h2e|

Pace is moving so fast that I can state-of-the-art research from just a few
years ago in my spare time.

|h2| Convolution or Components? |h2e|

Explain why it's hard with CNNs

* PCA (whitening didn't work)
* cnns didn't work
  * Point to reddit post

|h2| Implementation Notes |h2e|

Annotated notebook, but here are some of the (obvious to most people) points
that I figured

* batch norm, "RELU", dropout
* bigger batch size to speed up iterations (if gpu is big enough)
* epoch = 1000
  * Show graph of loss
* use entire dataset with keras fit_generator
* big network with regularization (link to CS291 course)
* Explain intuition about loss function with normal distribution + PCA, optimize big term first, and then go downward
* Keras backend functions operate on the entire tensor, need to use "axis=-1" to operate on non-batch_size dims
* Lower learning rate to 0.0001 (from default 0.001, 0.005)
  * (Sometimes) Blow up gradient otherwise

|h2| VAE SVNH Results |h2e|

* Fun part: generate images
* Randomly generated numbers
* Analogies 


|h2| Further Reading |h2e|

* Previous Posts: `variational autoencoders <link://slug/variational-autoencoders>`__
* Relevant Reddit Posts:  `Gaussian observation VAE <https://www.reddit.com/r/MachineLearning/comments/4eqifs/gaussian_observation_vae/>`__, `Results on SVHN with Vanilla VAE? <https://www.reddit.com/r/MachineLearning/comments/3wp5pc/results_on_svhn_with_vanilla_vae/>`__
* "Tutorial on Variational Autoencoders", Carl Doersch, https://arxiv.org/abs/1606.05908
* "Auto-Encoding Variational Bayes", Diederik P. Kingma, Max Welling, https://arxiv.org/abs/1312.6114 
* "Code for reproducing results of NIPS 2014 paper "Semi-Supervised Learning with Deep Generative Models", Diederik P. Kingma, https://github.com/dpkingma/nips14-ssl/
* The Street View House Numbers (SVHN) Dataset, http://ufldl.stanford.edu/housenumbers/
* CS231n: Convolutional Neural Networks for Visual Recognition, Stanford University, http://cs231n.github.io/neural-networks-1/
* PCA Whitening - UFLDL Tutorial, Stanford University, http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
  
|br|


