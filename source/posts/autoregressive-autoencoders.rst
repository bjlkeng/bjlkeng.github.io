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

* Image of autoencoder?
* Explain problem with them: no probabilistic interpretation

.. figure:: /images/autoencoder_structure.png
  :width: 550px
  :alt: Vanilla Autoencoder
  :align: center

  Figure 1: Vanilla Autoencoder (source: `Wikipedia <https://en.wikipedia.org/wiki/Autoencoder>`_)

|h2| Autoregressive Autoencoders |h2e|

* Explain AR property, show math
* Show picture from paper on autoregressive property

|h2| MADE Implementation |h2e|



|h3| Implementation Notes |h3e|

* Didn't

# Notes
# - Adding a direct (auto-regressive) connection between input/output seemed to make a huge difference (150 vs. < 100 loss)
# - Actually may have been a bug that caused it?
# - Got to be careful when coding up layers since getting indexes for selection exactly right is important
# - Random order didn't really generate any images that were recognizable

* Doing custom layers in Keras is so much nicer than using lower level tensorflow don't you think?

|h2| Conclusion |h2e|


|h2| Further Reading |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, and `Semi-supervised Learning with Variational Autoencoders <link://semi-supervised-learning-with-variational-autoencoders>`__
* "MADE: Masked Autoencoder for Distribution Estimation", Germain, Gregor, Murray, Larochelle, `ICML 2015 <https://arxiv.org/pdf/1502.03509.pdf>`_
* Wikipedia: `Autoencoder <https://en.wikipedia.org/wiki/Autoencoder>`_
* Github code for "MADE: Masked Autoencoder for Distribution Estimation", https://github.com/mgermain/MADE

