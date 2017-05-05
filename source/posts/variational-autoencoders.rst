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
generate data points) from a distribution similar to your observed data
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

Let's continue to use this handwritten digit generative model as our motivation
for a generative model.  Generating a 28x28 greyscale image that looks like a digit
is non-trivial, especially if we are trying to model it directly.  The joint
distribution over 28x28 random variables is going to be complex, for example,
enforcing that "0"s have empty space near the middle but "1"s don't, is not
very clear.  Typically in these situations, we'll introduce latent variables
which encode higher level ideas.  In our example, one of the latent variables
might correspond to which digit we're using (0-9), another one may be the
stroke width we use, and so on.  This model is simpler because there are
usually fewer parameters to estimate, reducing the number of data points
required for a good fit.  See my post on 
`the expectation-maximization algorithm <link://slug/the-expectation-maximization-algorithm>`__,
which has a brief description of latent variable models in the background section.

One downsides of any latent variable model is that you have to specify the 
model! That is, you have to have some idea of what latent variables you want
to include, how these variables are related to each other and the observed variables,
and finally how to fit the model (which depends on the connectivity).
All of the introduce potential for a misspecification of the model.  For
example, maybe you forgot to include stroke width and now all your handwritten
digits are blurry because it averaged over types of stroke widths in your
training dataset.  Wouldn't it be nice if you *didn't* need to explicitly
specify the latent variables (and associated distributions), nor the
relationships between them, and on top of all of this had an easy way to fit
the model?  Enter variational autoencoders.

|h3| From a Standard Normal Distributions to a Complex Latent Variable Model |h3e|

There are a couple of big ideas here that allow us to create this implicit model
without explicitly specifying anything.
The first big idea here is that we're not going to explicitly define any
latent variables, that is, we won't say "this variable is for 0-9 digit", 
"this variable is for stroke width".  Instead, we'll have our latent variables
as a simple uninterpretable standard multivariate normal distributions 
:math:`\mathcal{N}(0, I)` where :math:`I` is the identify matrix.  You may
be wondering how we can ever model anything complex if we just use a normal
distribution?  This leads us to the next big idea.

The second big idea is that starting from any random variable :math:`Z`, there
exists a *deterministic* function :math:`Y=g(Z)` such that :math:`Y` can be any
target distribution you want (See the box on "Inverse Transform Sampling"
below).  *The ingenious idea here is that we can learn* :math:`g(\cdot)` *from
the data*!  Thus, our variational autoencoder can transform our boring, old
normal distribution into any funky shaped distribution we want. 
As you may have already guessed, we use a neural network as a function
approximator to learn :math:`g(\cdot)`.

The last little bit in defining our latent variable model is translating our
latent variable into the final distribution of our observed data.  Here,
we'll also use something simple: we'll assume that the observed data
follows a normal distribution :math:`\mathcal{N}(g(z), \sigma^2 * I)`, with
mean following our learned latent random variable from the output of :math:`g`,
and identity covariance matrix scaled by a hyperparameter :math:`\sigma^2`.

The reason why we want to put a distribution on the output is that we want
to say that our output is *like* our observed data -- not exactly equal.
Remember, we're using a probabilistic interpretation here, so we need to write
a likelihood function and then maximize it usually by taking its derivative.
If we didn't have an output distribution, we would implicitly be saying that
:math:`g(z)` was exactly equal i.e. a Dirac delta function, which would result
in a discontinuity.  The is important because we will eventually want to use
gradient descent to learn :math:`g(z)` and this implicitly requires a smooth
function.  We'll see how this probabilistic interpretation plays into the
loss/objective function below.

.. admonition:: Inverse Transform Sampling

    `Inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`__
    is a method for sampling from any distribution given its cumulative
    distribution function (CDF), :math:`F(x)`.  It works as such:

    1. Sample a value, :math:`u`, between :math:`[0,1]` from a uniform
       distribution.
    2. Define the inverse of the CDF as :math:`F^{-1}(u)` (the domain is a 
       probability value between :math:`[0,1]`).
    3. :math:`F^{-1}(u)` is a sample from your target distribution.

    Of course, this method has no claims on being efficient.  For example,
    on continuous distributions, we would need to be able to find the inverse
    of the CDF (or some close approximation), which is not at all trivial.
    Typically, there are more efficient ways to perform sampling on any
    particular distribution but this provides a theoretical way to
    sample from *any* distribution.

    **Proof** 

    The proof of correctness is actually pretty simple.  Let :math:`U`
    be a uniform random variable on :math:`[0,1]`, and :math:`F^{-1}`
    as before, then we have:

    .. math::

        &P(F^{-1}(U) \leq x) \\
        &= P(U \leq F(x)) && \text{apply } F \text{ to both sides} \\
        &= F(x)  && \text{because } P(U\leq y) = y \\
        \tag{3}

    Thus, we have shown that :math:`F^{-1}(U)` has the distribution
    of our target random variable (since the CDF is the same).  
    
    It's important to note what we did: we took an easy to sample random
    variable :math:`U`, performed a *deterministic* transformation
    :math:`F^{-1}(U)` and ended up with a random variable that was distributed
    according to our target distribution.

    **Example** 

    As a simple example, we can try to generate a exponential distribution
    with CDF of :math:`F(x) = 1 - e^{-\lambda x}` for :math:`x \geq 0`.
    The inverse is defined by :math:`x = F^{-1}(u) = -\frac{1}{\lambda}\log(1-y)`.
    Thus, we can sample from an exponential distribution just by iteratively
    sampling evaluating this expression with a uniform randomly distributed
    number.

    .. figure:: /images/Inverse_transformation_method_for_exponential_distribution.jpg
      :height: 300px
      :alt: Visualization of mapping between a uniform distribution and an exponential one (source: Wikipedia)
      :align: center
    
      Figure 1: The :math:`y` axis is our uniform random distribution and the :math:`x` axis is our exponentially distributed number.  You can see for each point on the :math:`y` axis, we can map it to a point on the :math:`x` axis.  Even though :math:`y` is distributed uniformally, their mapping is concentrated on values closer to :math:`0` on the :math:`x` axis, matching an exponential distribution (source: Wikipedia).

    **Extensions** 

    Now instead of starting from a uniform distribution, what happens if we
    want to sample from another distribution, say a normal distribution? 
    We just first apply the reverse of the inverse sampling transform
    called the 
    `Probability Integral Transform <https://en.wikipedia.org/wiki/Probability_integral_transform>`__.
    So the steps would be:

    1. Sample from a normal distribution.
    2. Apply the probability integral transform using the CDF of a normal
       distribution to get a uniformly distributed sample.
    3. Apply inverse transform sampling with the inverse CDF of the target
       distribution to get a sample from our target distribution.

    What about extending to multiple dimensions?  We can just break up the
    joint distribution into its conditional components and sample each
    sequentially to construct the overall sample:

    .. math::

        P(x_1,\ldots, x_n) = P(x_n|x_{n-1}, \ldots,x_1)\ldots P(x_2|x_1)P(x_1) \tag{4}

    In detail, first sample :math:`x_1` using the method above, then :math:`x_2|x_1`,
    then :math:`x_3|x_2,x_1`, and so on.  Of course, this implicitly means you
    would have the CDF of each of those distributions available, which
    practically might not be possible.




.. figure:: /images/variational_autoencoder-decoder.png
  :height: 450px
  :alt: Variational Autoencoder Graphical Model
  :align: center

  Figure 2: A graphical model of a typical variational autoencoder (without a "encoder"). We're using a modified plate notation, the circles represent variables/parameters, rectangular boxes with to represent multiple instances of the contained variables, and the little diagram in the middle is a representation of a deterministic neural network (function approximator).

|br|

Writing the model out more formally, we have:

.. math::

    X_i &\sim \mathcal{N}(\mu_i, \sigma^2 * I) &&& \text{Observed variable} \\
    \mu_i &\sim g(Z_{1,\ldots,K}; \theta) &&& \text{Implicit latent variable}\\
    Z_k &\sim \mathcal{N}(0, I)

where:

* :math:`X_{i=1,\ldots,N}` are our normally distributed observations
* :math:`\mu_{i=1,\ldots,N}` are the mean of our observed variables which potentially has a very complex distribution
* :math:`\sigma^2` is a hyperparameter
* :math:`I` is the identify matrix
* :math:`g(Z_{1,\ldots,K}; \theta)` is a deterministic function with parameters to be learned :math:`\theta` (e.g. weights of a neural network)
* :math:`Z_{i=1,\ldots,K}` is a standard normally distributed random variable and our starting point for latent variables

Note: we can put another distribution on :math:`X` like a Bernoulli for binary
data parameterized by :math:`p=g(z;\theta)`.  The important part is we're able to
maximize the likelihood over the :math:`\theta` parameters.  Implicitly, we
will want our output variable to be continuous in :math:`\theta` so we can
perform gradient descent.

|h3| A hard fit |h3e|

Given the model above, we have all we need to fit the model: observations from
:math:`X`, well defined parameters for the latent variables, and a function
approximator :math:`g(\cdot)`, the only thing we need to find is :math:`\theta`.
This is a classic optimization problem where we could use an MLE (or MAP)
estimate.  Let's see how this works out.

First, we need to define the probability of seeing a single example :math:`x`:

.. math::

    P(X=x) &= \int p(X=x,Z=z) dz \\
           &= \int p(X=x|z;\theta)p(z) dz \\
    &= \int p_{\mathcal{N}}(x|g(z;\theta);\sigma^2*I)
            p_{\mathcal{N}}(z|0;I) dz \\
    &\approx \frac{1}{M} \sum_{m=1}^M p_{\mathcal{N}}(x|g(z_m;\theta);\sigma^2*I)
    &&& \text{where } z_m \sim N(0,I) \\
    \tag{5}

The probability of a single sample is just the joint probability of our given
model marginalizing (i.e. integrating) out :math:`Z`.  Since we don't have a 
analytical form of the density, we approximate the integral by averaging over
:math:`M` samples from :math:`Z\sim N(0, I)`.

Putting together the log-likelihood by logging the density and summing over all
of our :math:`N` observations:

.. math::

    P(X) = \frac{1}{M} \sum_{i=1}^N 
           \log(\sum_{m=1}^M p_{\mathcal{N}}(x_i|g(z_m;\theta);\sigma^2*I)) \tag{6}

Two problems here. First, the :math:`\log` can't be pushed inside the
summation, which actually isn't much a problem because we're not trying to
derive an analytical expression here, so long as we can use gradient descent
to learn :math:`\theta`, we're good.  In this case, we can easily take derivatives
since our density is normally distributed.

The other big problem, that is not easily seen through the notation, is that
:math:`z_m` is actually a :math:`K`-dimensional vector.  In order to approximate
the integral properly, we would have to sample over a *huge* number of
:math:`z` values!  This basically plays into the 
`curse of dimensionality <https://en.wikipedia.org/wiki/Curse_of_dimensionality>`__
whereby each additional dimension of :math:`z` exponentially increases the 
number of samples you need to properly approximate the volume of the space.
For small :math:`K` this might be feasible but any reasonable value will
intractable.




|h2| Variational Autoencoders |h2e|

- Variational autoencoders approximate the generative process
- Solidly based in probability
- Nothing to do with traditional auto-encoders

 

|h2| Deriving the Variational Autoencoder  |h2e|

- Explain why we need posterior P(z|X): help sample more efficiently
  from z, thus we don't need to integrate across it every time
- P(z|X) is intractable in general, introduce an approximation => variational inference
- Show the KL/var. Bayes equation
- Explain intuition of how instead of doing the full expecation E[log P(X|z)]


|h2| Further Reading |h2e|

* Previous Posts: 
  `Variational Bayes and The Mean-Field Approximation <link://slug/variational-bayes-and-the-mean-field-approximation>`__, `Variational Calculus <link://slug/the-calculus-of-variations>`__
* Wikipedia: `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Generative Models <https://en.wikipedia.org/wiki/Generative_model>`__, `Autoencoders <https://en.wikipedia.org/wiki/Autoencoder>`__, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__, `Inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`__
* "Tutorial on Variational Autoencoders", Carl Doersch, https://arxiv.org/abs/1606.05908
  
|br|
