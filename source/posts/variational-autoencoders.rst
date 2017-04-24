.. title: Variational Autoencoders
.. slug: variational-autoencoders
.. date: 2017-04-20 10:19:36 UTC-04:00
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

This post is going to talk about an incredibly interesting unsupervised
learning method in machine learning called variational autoencoders.  It's main
claim to fame is in building generative models of complex distributions like
handwritten digits, faces, image segmentation among others.  The really cool
thing about this topic is that it has firm roots in probability but uses a
function approximator (i.e.  neural networks) to approximate an otherwise
intractable problem.  As usual, I'll try to start with some background and
motivation, include a healthy does of math, and along the way try to convey
some of the intuition of why it works.  I'll also show a bit of code and point
you to some examples for you to try yourself.  I based much of this post on
Carl Doersch's `tutorial <https://arxiv.org/abs/1606.05908>`__, which has a
great explanation on this whole topic, so make sure you check that out too.

.. TEASER_END

|h2| Generative Models  |h2e|

The first place on this topic is to discuss the idea of a 
`generative model <https://en.wikipedia.org/wiki/Generative_model>`__.
A generative model is a model from which allows you to sample (i.e. randomly
generate data points) from the a distribution similar to your observed data
points.  We can accomplish this by specifying a joint distribution over 
all the dimensions of the data (including the "y" labels).
This allows us to generate any number of data points that has similar
characteristics to our observed data.  This is in contrast to a   
`discriminative model <https://en.wikipedia.org/wiki/Discriminative_model>`__,
which only "generates" your target outcome variable.  For example, a binary
classifier only outputs 0 or 1 "y" labels and cannot generate a data point that
looks like your "X" features.

Typically, as part of your model you'll want to specify latent variables that
represent some higher level concept.  For example, when modelling housing
prices you may want to have a latent variable for the neighborhood.  This
allows for complex relationships between the latent variables and the observed
ones.  We'll be focusing on the application of generating new values that look
like the observation but check out the 
`Wikipedia <https://en.wikipedia.org/wiki/Generative_model>`__ article for a
better picture of some other applications.

.. admonition:: Example 1: Generative Models

    1. **Normal Distribution** for modeling human heights:
        Although generative models are usually only talked about in the context
        of complex latent variable models, technically, a simple probability
        distribution is a also generative model.  In this example, if we have
        the height of different people (:math:`x`) as our 1-dimensional
        observations, we can have a normal distribution as our generative model:

        .. math::

            x &\sim \mathcal{N}(\mu, \sigma^{2}) \\
            \tag{1}
        
        With this simple model, we can "generate" heights that look like our
        observations by sampling the normal distribution (after fitting the
        :math:`\mu` and :math:`\sigma^2` parameters).  Any data point we sample
        from this distribution could plausibly be another observation we had
        (assuming our model was a good fit for the data).
        
    2. **Gaussian Mixture Model** for modelling prices of different houses:
        Let's suppose we have :math:`N` observations of housing prices in a
        city.  We hypothesize that within a particular neighbourhood, house
        prices tend to cluster around the neighborhood mean.  Thus, we can
        model this situation using a Gaussian mixture model as such:
        
        .. math::

            p(x_i|\theta) &=  \sum_{k=1}^K p(z_i=k) p(x_i| z_i=k, \mu_k, \sigma_k^2)  \\
            x_i| z_i &\sim \mathcal{N}(\mu_k, \sigma_k^2) \\
            z_i &\sim \text{Categorical}(\pi) \tag{2}
        
        where :math:`z_i` is a categorical variable for a given neighbourhood,
        :math:`x_i|z_i` is a normal distribution for the prices within a given
        neighborhood :math:`k`, and :math:`x_i` will be a Gaussian mixture of
        each of the component neighborhoods.

        Using this model, we could then generate several different types of 
        observations.  If we wanted to generate a house of a particular
        neighborhood, we could sample theo normal distribution from
        :math:`x_i|z_i`.  If we wanted to sample the "average" house, we could
        sample a price from each neighborhood, and then compute their weighted
        average in proportion to the distribution of the categorical variable
        :math:`z_i`.  
        
        This more complex model is not as straighforward to fit.  A common
        method is to use
        `the expectation-maximization algorithm <link://slug/the-expectation-maximization-algorithm>`__ or something similar such as variational inference.
        
    3. **Handwritten digits**:
        A more modern application of generative models is for a hand written
        digits.  The `MNIST database
        <https://en.wikipedia.org/wiki/MNIST_database>`__ is a large dataset of
        handwritten digits typically used as a benchmark for machine learning
        techniques.  Many recent techniques have shown good performance in
        generating new samples of hand written digits.  Two of the most popular
        approaches are variational autoencoders (the topic of this post) and
        `generative adversarial networks
        <https://en.wikipedia.org/wiki/Generative_adversarial_networks>`__.

        In these types of approaches, the generative model is trained on the
        tens of thousands of examples of 28x28 greyscale images, each
        representing a single "0" to "9" digit.  Once trained, the model should
        be able to reproduce random "0" to "9" 28x28 greyscale digits that, if
        trained well, look like a hand written digit.
        

|h2| An Implicit Generative Model |h2e|

- previous section we explicitly specified latent variables e.g. neighborhoods.
- this is useful for certain applications
- but what if we didn't care about actually interpretations of the latent variables
  and instead just wanted the best model to generate observations
- this makes sense because most likely we wouldn't be able to specify *everything*,
  for hand written digit example, we would need things like style, slant, etc.
  couldn't specify everything
- what if instead, we "learned" those things implicitly?

- Explain how any RV can be generated by a base set of normal distributions + 
  a deterministic function
- Show equations for generative models, including latent variables.
- Explain problems with latent variables: have to specify network, often hard
  to train

|h2| Variational Autoencoders |h2e|

- Variational autoencoders approximate the generative process
- Solidly based in probability
- Nothing to do with traditional auto-encoders

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

|h2| Deriving the Variational Autoencoder  |h2e|

- Explain why we need posterior P(z|X): help sample more efficiently
  from z, thus we don't need to integrate across it every time
- P(z|X) is intractable in general, introduce an approximation => variational inference
- Show the KL/var. Bayes equation
- Explain intuition of how instead of doing the full expecation E[log P(X|z)]


|h2| Further Reading |h2e|

* Previous Posts: 
  `Variational Bayes and The Mean-Field Approximation <link://slug/variational-bayes-and-the-mean-field-approximation>`__, `Variational Calculus <link://slug/the-calculus-of-variations>`__
* Wikipedia: `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Generative Models <https://en.wikipedia.org/wiki/Generative_model>`__, `Autoencoders <https://en.wikipedia.org/wiki/Autoencoder>`__, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__
* "Tutorial on Variational Autoencoders", Carl Doersch, https://arxiv.org/abs/1606.05908
  
|br|
