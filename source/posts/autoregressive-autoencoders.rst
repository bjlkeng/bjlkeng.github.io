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



Write your post here.


|h2| Vanilla Autoencoders |h2e|

* Image of autoencoder?
* Explain problem with them: no probabilistic interpretation


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

|h2| Conclusion |h2e|


|h2| Further Reading |h2e|

* "MADE: Masked Autoencoder for Distribution Estimation", Germain, Gregor, Murray, Larochelle, `ICML 2015 <https://arxiv.org/pdf/1502.03509.pdf>`_
* Wikipedia: `Autoencoder <https://en.wikipedia.org/wiki/Autoencoder>`_
* Github code for "MADE: Masked Autoencoder for Distribution Estimation", https://github.com/mgermain/MADE

