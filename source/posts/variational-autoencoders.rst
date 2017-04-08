.. title: Variational Autoencoders
.. slug: variational-autoencoders
.. date: 2017-04-07 10:19:36 UTC-04:00
.. tags: variational calculus, autoencoders, Kullback-Leibler, generative models, mathjax
.. category: 
.. link: 
.. description: A brief introduction into variational autoencoders.
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

.. TEASER_END

|h2| Generative Models  |h2e|

- Show equations for generative models, including latent variables.
- Explain problems with latent variables: have to specify network, often hard
  to train

|h2| Variational Autoencoders |h2e|

- Variational autoencoders approximate the generative process
- Solidly based in probability
- Not exactly anything to do with traditional auto-encoders

|h3| An Unusual Approach to Latent Variables |h3e|

- Explain N(0,1) latent variables, and use a powerful deterministic function
  to get to the actual distribution you want
- Show a visualization of latent variables going into X=f(z;\theta)
- Explain why it's hard to train: If we have a large latent space (Z_1, ... Z_k),
  we have to sample 100^k to get a approximation of the whole space.  Most of
  this space is such a low probability contribution to finding P(X|z) that it's wasteful
  to spend time searching. (Curse of dimensionality)
- Question: Can we find a more directed method to sample latent space so that we can
  more directly optimize for high probability values so we can converge faster?
- Well if we knew the distribution of z given a particular X, we could just sample
  the z's that we need.  This in fact is P(z|X) the posterior.
- See this in next section

Summary:

- 

|h2| Setting up the Variational Autoencoder  |h2e|



|h2| Further Reading |h2e|

* Previous Posts: 
  `Variational Bayes and The Mean-Field Approximation <link://slug/variational-bayes-and-the-mean-field-approximation>`__, `Variational Calculus <link://slug/the-calculus-of-variations>`__
* Wikipedia: `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Generative Models <https://en.wikipedia.org/wiki/Generative_model>`__, `Autoencoders <https://en.wikipedia.org/wiki/Autoencoder>`__, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__
* "Tutorial on Variational Autoencoders", Carl Doersch, https://arxiv.org/abs/1606.05908
  

`A Beginner's Guide to Variational Methods: Mean-Field Approximation <http://blog.evjang.com/2016/08/variational-bayes.html>`__, Eric Jung.

|br|


