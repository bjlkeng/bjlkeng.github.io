.. title: Variational Autoencoders with Inverse Autoregressive Flows
.. slug: variational-autoencoders-with-inverse-autoregressive-flows
.. date: 2017-12-04 07:47:38 UTC-05:00
.. tags: variational calculus, autoencoders, Kullback-Leibler, generative models, MNIST, autoregressive, MADE, mathjax
.. category: 
.. link: 
.. description: An introduction to normalizing flows and inverse autoregressive flows for variational inference.
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

(YAAP: Yet another autoencoder post) In this post, I'm going to be
describing a really cool idea about how to improve variational
autoencoders using inverse autoregressive flows.  The main idea is
that we can generate more powerful posterior distributions compared to
the vanilla diagonal Gaussians by applying a series of invertible
transformations.  This, in theory, will allow your variational
autoencoder to fit better by concentrating the stochastic samples
around a closer approximation to the true posterior.  The math works
out so nicely while the results are kind of marginal [1]_.

.. TEASER_END

|h2| Normalizing Flows |h2e|

* transformation of probability density under a 1-1 function (invertible): `Probability Density Function: Dependent variables and change of variables <https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables>`__
* Show the likelihood
* Call out box for transforming variables?
* Applied to posterior of variational inference [1]

|h2| Inverse Autoregressive Transformations |h2e|

* Autoregressive property: `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__, [2]
* Autoregressive equations
* Show derivation of inverse autoregressive equations (the Gaussian form)
* Diagram showing forward and backwards

|h2| Inverse Autoregressive Flows |h2e|

* Show block diagram from paper [3]
  * Context 'h'
* Jacobians are triangular wrt to dmu/dz_{t-1}
* Show a VAE diagram
* Explain stable computation sigma * z + (1-sigma) * m

|h3| Experiments: IAF Implementation |h3e|

TODO

|h3| Implementation Notes |h3e|

* The "context" vector is connected to the input of the MADE with additional dense layer (see impl.)
* I added a 2 * to the stability computation, seems like you will need it or else you can't "expand" the volume
* Had a lot of trouble with getting things to be stable with made computation, not sure if all of them helped:
    * Added regularizers on the autoregressive parts
    * Used 'sigmoid' for activation for all made stuff (instead of 'elu' for others)
* Had a bunch of confusion with the logqz_x computation, in particular the determinant.  Only after I worked through the math did I actually figure out the sign of the determinant in Eq. X
* 

|h2| Conclusion |h2e|

|h2| Further Reading |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, `Semi-supervised Learning with Variational Autoencoders <link://slug/semi-supervised-learning-with-variational-autoencoders>`__, `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__
* My implementation on Github:
* [1] "Variational Inference with Normalizing Flows", Danilo Jimenez Rezende, Shakir Mohamed, `ICML 2015 <https://arxiv.org/abs/1505.05770>`__
* [2] "MADE: Masked Autoencoder for Distribution Estimation", Germain, Gregor, Murray, Larochelle, `ICML 2015 <https://arxiv.org/pdf/1502.03509.pdf>`__
* [3] "Improving Variational Inference with Inverse Autoregressive Flow", Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling, `NIPS 2016 <https://arxiv.org/abs/1606.04934>`_
* Wikipedia: `Probability Density Function: Dependent variables and change of variables <https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables>`__
* Github code for "Improving Variational Inference with Inverse Autoregressive Flow": https://github.com/openai/iaf/

.. [1] At least by my estimate the results are kind of marginal. Improving the posterior on its own doesn't seem to have a significant boost in the likelihood.  The IAF paper [3] actually does have really good results on CIFAR10 but uses a novel architecture combined with IAF transforms.  So by itself, the IAF doesn't do that much.
