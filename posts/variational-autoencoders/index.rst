.. title: Variational Autoencoders
.. slug: variational-autoencoders
.. date: 2017-05-30 8:19:36 UTC-04:00
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
handwritten digits, faces, and image segments among others.  The really cool
thing about this topic is that it has firm roots in probability but uses a
function approximator (i.e.  neural networks) to approximate an otherwise
intractable problem.  As usual, I'll try to start with some background and
motivation, include a healthy does of math, and along the way try to convey
some of the intuition of why it works.  I've also annotated a 
`basic example <https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational-autoencoder.ipynb>`__ 
so you can see how the math relates to an actual implementation.  I based much
of this post on Carl Doersch's `tutorial <https://arxiv.org/abs/1606.05908>`__,
which has a great explanation on this whole topic, so make sure you check that
out too.

.. TEASER_END

|h2| 1. Generative Models  |h2e|

The first thing to discuss is the idea of a 
`generative model <https://en.wikipedia.org/wiki/Generative_model>`__.
A generative model is a model which allows you to sample (i.e. randomly
generate data points) from a distribution similar to your observed (i.e. training)
data.  We can accomplish this by specifying a joint distribution over 
all the dimensions of the data (including the "y" labels).
This allows us to generate any number of data points that has similar
characteristics to our observed data.  This is in contrast to a   
`discriminative model <https://en.wikipedia.org/wiki/Discriminative_model>`__,
which can only model dependence between the outcome variable (:math:`y`) and
features (:math:`X`).  For example, a binary classifier only outputs 0 or 1 "y"
labels and cannot generate a data point that looks like your "X" features.

Typically, as part of your model you'll want to specify latent variables that
represent some higher level concepts.  This allows for intuitive relationships
between the latent variables and the observed ones, while usually simplifying
the overall number of parameters (i.e. complexity) of the model.  We'll be
focusing on the application of generating new values that look like the
observed data but check out the 
`Wikipedia <https://en.wikipedia.org/wiki/Generative_model>`__ 
article for a better picture of some other applications.

.. admonition:: Example 1: Generative Models

    1. **Normal Distribution** for modelling human heights:
        Although generative models are usually only talked about in the context
        of complex latent variable models, technically, a simple probability
        distribution is a also generative model.  In this example, if we have
        the height of different people (:math:`x`) as our 1-dimensional
        observations, we can use a normal distribution as our generative model:

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
        prices tend to cluster around the neighbourhood mean.  Thus, we can
        model this situation using a Gaussian mixture model as such:
        
        .. math::

            p(x_i|\theta) &=  \sum_{k=1}^K p(z_i=k) p(x_i| z_i=k, \mu_k, \sigma_k^2)  \\
            x_i| z_i &\sim \mathcal{N}(\mu_k, \sigma_k^2) \\
            z_i &\sim \text{Categorical}(\pi) \tag{2}
        
        where :math:`z_i` is a categorical variable for a given neighbourhood,
        :math:`x_i|z_i` is a normal distribution for the prices within a given
        neighbourhood :math:`z_i=k`, and :math:`x_i` will be a Gaussian mixture of
        each of the component neighbourhoods.

        Using this model, we could then generate several different types of 
        observations.  If we wanted to generate a house of a particular
        neighbourhood, we could sample the normal distribution from
        :math:`x_i|z_i`.  If we wanted to sample the "average" house, we could
        sample a price from each neighbourhood, and then compute their weighted
        average in proportion to the distribution of the categorical variable
        :math:`z_i`.  
        
        This more complex model is not as straightforward to fit.  A common
        method is to use
        `the expectation-maximization algorithm <link://slug/the-expectation-maximization-algorithm>`__ or something similar such as variational inference.
        
    3. **Handwritten digits**:
        A more modern application of generative models is for hand written
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
        

|h2| 2. An Implicit Generative Model (aka the "decoder") |h2e|

Let's continue to use this handwritten digit as our motivation
for generative models.  Generating a 28x28 greyscale image that looks like a digit
is non-trivial, especially if we are trying to model it directly.  The joint
distribution over 28x28 random variables is going to be complex, for example,
enforcing that "0"s have empty space near the middle but "1"s don't, is not
an easy thing to do.  Typically in these situations, we'll introduce latent variables
which encode higher level ideas.  In our example, one of the latent variables
might correspond to which digit we're using (0-9), another one may be the
stroke width we use, etc.  This model is simpler because there are
usually fewer parameters to estimate, reducing the number of data points
required for a good fit.  (See my post on 
`the expectation-maximization algorithm <link://slug/the-expectation-maximization-algorithm>`__,
which has a brief description of latent variable models in the background section)

One of the downsides of any latent variable model is that you have to specify the 
model! That is, you have to have some idea of what latent variables you want
to include, how these variables are related to each other and the observed variables,
and finally how to fit the model (which depends on the connectivity).
All of these issues introduce potential for a misspecification of the model.  For
example, maybe you forgot to include stroke width and now all your handwritten
digits are blurry because it averaged over all the types of stroke widths in your
training dataset.  Wouldn't it be nice if you didn't need to explicitly
specify the latent variables (and associated distributions), nor the
relationships between them, and on top of all of this had an easy way to fit
the model?  
Enter variational autoencoders.

|h3| 2.1 From a Standard Normal Distributions to a Complex Latent Variable Model |h3e|

There are a couple of big ideas here that allow us to create this implicit model
without explicitly specifying anything.
The first big idea here is that we're not going to explicitly define any
latent variables, that is, we won't say "this variable is for digit 0", "this one for digit 1", 
...,  "this variable is for stroke width", etc.  Instead, we'll have our latent variables
as a simple uninterpretable standard `isotropic
<https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic>`__
multivariate normal distribution :math:`\mathcal{N}(0, I)` where :math:`I` is
the identify matrix.  You may be wondering how we can ever model anything
complex if we just use a normal distribution?  This leads us to the next big
idea.

The second big idea is that starting from any random variable :math:`Z`, there
exists a *deterministic* function :math:`Y=g(Z)` (under most conditions) such
that :math:`Y` can be any complex target distribution you want (see the box on "Inverse
Transform Sampling" below).  *The ingenious idea here is that we can learn*
:math:`g(\cdot)` *from the data*!  Thus, our variational autoencoder can
transform our boring, old normal distribution into any funky shaped
distribution we want!  As you may have already guessed, we use a neural network
as a function approximator to learn :math:`g(\cdot)`.

The last little bit in defining our latent variable model is translating our
latent variable into the final distribution of our observed data.  Here,
we'll also use something simple: we'll assume that the observed data
follows a isotropic normal distribution :math:`\mathcal{N}(g(z), \sigma^2 * I)`, with
mean following our learned latent random variable from the output of :math:`g`,
and identity covariance matrix scaled by a hyperparameter :math:`\sigma^2`.

The reason why we want to put a distribution on the output is that we want
to say that our output is *like* our observed data -- not exactly equal.
Remember, we're using a probabilistic interpretation here, so we need to write
a likelihood function and then maximize it, usually by taking its gradient.
If we didn't have an output distribution, we would implicitly be saying that
:math:`g(z)` was exactly equal i.e. a Dirac delta function, which would result
in a discontinuity.  This is important because we will eventually want to use
stochastic gradient descent to learn :math:`g(z)` and this implicitly requires
a smooth function.  We'll see how this probabilistic interpretation plays into
the loss/objective function below.

.. admonition:: Inverse Transform Sampling

    `Inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`__
    is a method for sampling from any distribution given its cumulative
    distribution function (CDF), :math:`F(x)`. 
    For a given distribution with CDF :math:`F(x)`, it works as such:

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
        &= F(x)  && \text{because } P(U\leq y) = y \text{ on } [0,1] \\
        \tag{3}

    Thus, we have shown that :math:`F^{-1}(U)` has the distribution
    of our target random variable (since the CDF :math:`F(x)` is the same).  
    
    It's important to note what we did: we took an easy to sample random
    variable :math:`U`, performed a *deterministic* transformation
    :math:`F^{-1}(U)` and ended up with a random variable that was distributed
    according to our target distribution.

    **Example** 

    As a simple example, we can try to generate a exponential distribution
    with CDF of :math:`F(x) = 1 - e^{-\lambda x}` for :math:`x \geq 0`.
    The inverse is defined by :math:`x = F^{-1}(u) = -\frac{1}{\lambda}\log(1-y)`.
    Thus, we can sample from an exponential distribution just by iteratively
    evaluating this expression with a uniform randomly distributed number.

    .. figure:: /images/Inverse_transformation_method_for_exponential_distribution.jpg
      :height: 300px
      :alt: Visualization of mapping between a uniform distribution and an exponential one (source: Wikipedia)
      :align: center
    
      Figure 1: The :math:`y` axis is our uniform random distribution and the :math:`x` axis is our exponentially distributed number.  You can see for each point on the :math:`y` axis, we can map it to a point on the :math:`x` axis.  Even though :math:`y` is distributed uniformly, their mapping is concentrated on values closer to :math:`0` on the :math:`x` axis, matching an exponential distribution (source: Wikipedia).

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
  :height: 400px
  :alt: Variational Autoencoder Graphical Model
  :align: center

  Figure 2: A graphical model of a typical variational autoencoder (without a "encoder", just the "decoder"). We're using a modified plate notation: the circles represent variables/parameters, rectangular boxes with a number in the lower right corner to represent multiple instances of the contained variables, and the little diagram in the middle is a representation of a deterministic neural network (function approximator).

|br|

Writing the model out more formally, we have:

.. math::

    X_i &\sim \mathcal{N}(\mu_i, \sigma^2 * I) &&& \text{Observed variable} \\
    \mu_i &\sim g(Z_{1,\ldots,K}; \theta) &&& \text{Implicit latent variable}\\
    Z_k &\sim \mathcal{N}(0, I) 
    \tag{5}

where:

* :math:`X_{i=1,\ldots,N}` are our normally distributed observations
* :math:`\mu_{i=1,\ldots,N}` are the mean of our observed variables which potentially has a very complex distribution
* :math:`\sigma^2` is a hyperparameter
* :math:`I` is the identify matrix
* :math:`g(Z_{1,\ldots,K}; \theta)` is a deterministic function with parameters to be learned :math:`\theta` (e.g. weights of a neural network)
* :math:`Z_{i=1,\ldots,K}` are standard isotropic normally distributed random variables

Note: we can put another distribution on :math:`X` like a Bernoulli for binary
data parameterized by :math:`p=g(z;\theta)`.  The important part is we're able to
maximize the likelihood over the :math:`\theta` parameters.  Implicitly, we
will want our output variable to be continuous in :math:`\theta` so we can
take its gradient.

|h3| 2.2 A hard fit |h3e|

Given the model above, we have all we need to fit the model: observations from
:math:`X`, fully defined latent variables (:math:`Z`), and a
function approximator :math:`g(\cdot)`; the only thing we need to find is
:math:`\theta`.  This is a classic optimization problem where we could use an
MLE (or MAP) estimate.  Let's see how this works out.

First, we need to define the probability of seeing a single example :math:`x`:

.. math::

    P(X=x) &= \int p(X=x,Z=z) dz \\
           &= \int p(X=x|z;\theta)p(z) dz \\
    &= \int p_{\mathcal{N}}(x;g(z;\theta),\sigma^2*I)
            p_{\mathcal{N}}(z;0,I) dz \\
    &\approx \frac{1}{M} \sum_{m=1}^M p_{\mathcal{N}}(x;g(z_m;\theta),\sigma^2*I)
    &&& \text{where } z_m \sim \mathcal{N}(0,I) \\
    \tag{6}

The probability of a single sample is just the joint probability of our given
model marginalizing (i.e. integrating) out :math:`Z`.  Since we don't have an
analytical form of the density, we approximate the integral by averaging over
:math:`M` samples from :math:`Z\sim \mathcal{N}(0, I)`.

Putting together the log-likelihood (defined by log'ing the density and summing over all
of our :math:`N` observations):

.. math::

    \log P(X) \approx \frac{1}{N} \sum_{i=1}^N 
           \log(\frac{1}{M} \sum_{m=1}^M p_{\mathcal{N}}(x_i;g(z_m;\theta),\sigma^2*I)) \tag{7}

Two problems here. First, the :math:`\log` can't be pushed inside the
summation, which actually isn't much a problem because we're not trying to
derive an analytical expression here; so long as we can use gradient descent
to learn :math:`\theta`, we're good.  In this case, we can easily take derivatives
since our density is normally distributed.

The other big problem, that is not easily seen through the notation, is that
:math:`z_m` is actually a :math:`K`-dimensional vector.  In order to approximate
the integral properly, we would have to sample over a *huge* number of
:math:`z` values for *each* :math:`x_i` sample!  This basically plays into the 
`curse of dimensionality <https://en.wikipedia.org/wiki/Curse_of_dimensionality>`__
whereby each additional dimension of :math:`z` exponentially increases the 
number of samples you need to properly approximate the volume of the space.
For small :math:`K` this might be feasible but any reasonable value, it will be
intractable.

Looking at it from another point of view, the reason why this is intractable is because
it's inefficient; for each :math:`x`, we have to average over a large number
(:math:`M`) samples.  But, for any given observation :math:`x`, most of the
:math:`z_m` will contribute very little to the likelihood.
Using the handwritten digit example, if we're trying to generate a "0", most of
the values of :math:`z_m` sampled from our prior will have a very small
probability of generating a "0", so we're wasting a lot of computation in
trying to average over all these samples [1]_.

Wouldn't it be nice if we could just sample from :math:`p(z|X=x_i)` directly
and only pick :math:`z` values that contribute a significant amount to the
likelihood, thus getting rid the need for the large inefficient summation over
:math:`M` samples?  This is exactly what variational autoencoders proposes!

(Note: Just to be clear, each :math:`x_i` will likely have a *different*
:math:`p(z|X=x_i)`.  Imagine our hand written digit example, a "1" will
probably have a very different posterior shape than an "8".)

|h3| 2.2 Summary |h3e|

To summarize, this is what we're trying to accomplish:

* Our generative model is an implicit latent variable model with latent
  variables :math:`Z` as standard isotropic multivariate normal distribution.
* The :math:`Z` variables are transformed into an arbitrarily complex
  distribution by a deterministic function approximator (e.g. neural
  network parameterized by :math:`\theta`) that can model our data.
* We can fit our generative model via the likelihood by averaging over a huge
  number of :math:`Z` samples; this becomes intractable for higher dimensions
  (curse of dimensionality).
* For any given sample :math:`x_i`, most of the :math:`z` samples will
  contribute very little to the likelihood calculation, so we wish to sample
  only the probable values of :math:`z_m` that contribute significantly to
  the likelihood using the posterior distribution :math:`z|X=x_i`.

From here, we can finally get to the "variational" part of variational
autoencoders.

|h2| 3. Variational Bayes for the Posterior (aka the "encoder") |h2e|

From our novel idea of an implicit generative model, we come to a new problem:
how can we estimate the posterior distribution :math:`Z|X=x_i`? We have a couple of 
problems, first, the posterior probably has no closed form analytic solution.
This is not terrible because this is a typical problem in Bayesian inference
which we solve via either `Markov Chain Monte Carlo Methods <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__ or 
`Variational Bayes <link://slug/variational-bayes-and-the-mean-field-approximation>`__.
Second, we wanted to use the posterior :math:`Z|X=x_i` to maximize our
likelihood function :math:`P(X|Z;\theta)`,
but surely to find :math:`Z|X=x_i` we need to know :math:`X|Z` --
a circular dependence.  The solution to this is also novel, let's
*simultaneously*, fit our posterior, generate samples from it, *and* maximize
our original log-likelihood function!  

First, we'll attempt to solve the first problem of finding the posterior
:math:`P(z|X=x_i)` and this will lead us to the solution to the second problem
of fitting our likelihood.  Let's dig into some math to see how this works.

|h3| 3.1 Setting up the Variational Objective |h3e|

Since the posterior is so complex, we won't try to model it exactly.  Instead, 
we'll use variational Bayesian methods (see my 
`previous post <link://slug/variational-bayes-and-the-mean-field-approximation>`__)
to approximate it.  We'll denote our approximate distribution as
:math:`Q(Z|X)` and, as usual, we'll use 
`KL divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__ 
as our measure of "closeness" to the actual distribution.
Writing the math down, we get:

.. math::

    \mathcal{D}(Q(z|X) || P(z|X)) &= \int Q(z|X) \log\big[\frac{Q(z|X)}{P(z|X)}\big] dz \\ 
    &= E_{z\sim Q}[\log Q(z|X) - \log P(z|X)] \\
    &= E_{z\sim Q}[\log Q(z|X) + \log P(X) - \log P(X|z) - \log P(z)] &&& \text{Bayes rule} \\
    &= E_{z\sim Q}[\log Q(z|X) - \log P(X|z) - \log P(z)] + \log P(X)
    \tag{8}

We pulled :math:`P(X)` out of the expectation since it is not dependent on
:math:`z`.  Rearranging Equation 8:

.. math::

    \log P(X) - \mathcal{D}(Q(z|X) || P(z|X)) = 
        E_{z\sim Q}[\log P(X|z)] - \mathcal{D}(Q(z|X) || P(z)) \tag{9}

where we use the definition of KL divergence on the RHS.
This is the core objective of variational autoencoders for which we wish to
maximize.  The LHS defines our higher level goal:

* :math:`\log P(X)`: maximize the log-likelihood (Equation 7) 
* :math:`\mathcal{D}(Q(z|X) || P(z|X))`: Minimize the KL divergence between our
  approximate posterior :math:`Q(z|X)` to fit :math:`P(z|X)`

The RHS gives us an explicit objective where we know all the pieces, and we can
maximize via gradient descent (details in the next section):

* :math:`P(X|z)`: original implicit generative model
* :math:`Q(z|X)`: approximate posterior (we'll define what this looks like below)
* :math:`P(z)`: prior distribution of latent variables (standard isotropic normal distribution)

Notice, we now have what appears to be an "autoencoder".  :math:`Q(z|X)`
"encodes" :math:`X` to latent representation :math:`z`, and :math:`P(X|z)`
"decodes" :math:`z` to reconstruct :math:`X`.  We'll see how these equations
translate into a model that we can directly optimize.

|h3| 3.2 Defining the Variational Autoencoder Model |h3e|

Now that we have a theoretical framework that gives us an objective to optimize,
we need to explicitly define :math:`Q(z|X)`.  In the same way we implicitly
defined the generative model, we'll use the same idea to define the approximate
posterior.  We'll assume that the `Q(z|X)` is normally distributed with
mean and co-variance matrix defined by a neural network with parameters
:math:`\phi`.  The co-variance matrix is usually constrained to be diagonal to
simplify things a bit.  Formally, we'll have:

.. math::

    z|X &\approx \mathcal{N}(\mu_{z|X}, \Sigma_{z|X}) \\
    \mu_{z|X}, \Sigma_{z|X} &= g_{z|X}(X; \phi) \\
    \tag{10}

where:

* :math:`z|X` is our approximated posterior distribution as a multivariate
  normal distribution
* :math:`\mu_{z|X}` is a vector of means for our normal distribution
* :math:`\Sigma_{z|X}` is a diagonal co-variance matrix for our normal distribution
* :math:`g_{z|X}` is our function approximator (neural network)
* :math:`\phi` are the parameters to :math:`g_{z|X}`

As a first attempt, we might try to put Equations 5 and 10 to form a model
as shown in Figure 3.  The red boxes show our variational loss (objective)
function from Equation 9 (Note: the squared error comes from the :math:`\log`
of the normal distribution and `KL divergence between two multivariate normals
<https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Kullback.E2.80.93Leibler_divergence_for_multivariate_normal_distributions>`__
is well known); the input and the sampling of the :math:`z` values
is shown in blue. 


.. figure:: /images/variational_autoencoder2.png
  :width: 550px
  :alt: Variational Autoencoder Diagram
  :align: center

  Figure 3: An initial attempt at a variational autoencoder without
  the "reparameterization trick".  Objective functions shown in red.
  We cannot back-propagate through the stochastic sampling operation
  because it is not a continuous deterministic function.
  
Using this model, we can perform a "forward pass":

1. Inputting values of :math:`X=x_i`
2. Computing :math:`\mu_{z|X}` and :math:`\Sigma_{z|X}` from :math:`g_{z|X}(X;\phi)`
3. Sample a :math:`z` value from :math:`\mathcal{N}(\mu_{z|X}, \Sigma_{z|X})`
4. Compute :math:`\mu_{X|z}` from :math:`g_{X|z}(z;\theta)` to produce the
   (mean of the) reconstructed output.

Remember, our goal is to learn the parameters for our two networks: :math:`\theta`
and :math:`\phi`.  We can attempt to do this through back-propagation but
we hit a road-block when we back-propagate the :math:`||X-\mu_{X|z}||^2` error
through the sampling operation.  Since the sampling operation isn't a
continuous deterministic function, it has no gradient, thus we can't do any
form of back-propagation.  Fortunately, we can use the "reparameterization
trick" to circumvent this sampling problem.  This is shown in Figure 4.

.. figure:: /images/variational_autoencoder3.png
  :width: 550px
  :alt: Variational Autoencoder Diagram
  :align: center

  Figure 4: A variational autoencoder with the "reparameterization trick".
  Notice that all operations between the inputs and objectives are
  continuous deterministic functions, allowing back-propagation to occur.

The key insight is that :math:`\mathcal{N}(\mu_{z|X}, \Sigma_{z|X})`
is equivalent to :math:`\mathcal{N}(0, I) * \Sigma_{z|X} + \mu_{z|X}`.
That is, we can just sample from an standard isotropic normal distribution and then
scale and shift our sample to transform it to a sample from our desired
:math:`z|X` distribution.  Shifting the sampling operation to an input
of our model (for each sample :math:`x_i`, we can sample an arbitrary
number of :math:`\mathcal{N}(0,I)` samples to pair with it).

With the "reparameterization trick", this new model allows us to 
take the gradient of the loss function with respect to the target parameters 
(:math:`\theta` and :math:`\phi`).  In particular, we would want to take 
these gradients:

.. math::

    &\frac{\partial}{\partial\theta}(X-\mu_{X|z})^2 \\
    &\frac{\partial}{\partial\phi}(X-\mu_{X|z})^2 \\
    &\frac{\partial}{\partial\phi} \mathcal{D}(Q(z|X) || P(z)) \\
    \tag{11}

and iteratively update the weights of our neural network using
back-propagation.  As seen in Figure 4, :math:`Q(z|X) = N(\mu_{z|X},
\Sigma_{z|X})` is parameterized by :math:`\phi` via :math:`g_{z|X}(X;\phi)`,
whereas :math:`\mu_{X|z}` is implicitly parameterized by both :math:`\theta`
through via :math:`g_{X|z}(z;\theta)`, and :math:`\phi` through
:math:`z` and :math:`g_{z|X}(X;\phi)`.

|h3| 3.3 Training a Variational Autoencoder |h3e|

Now that we have the basic structure of our network and understand the relationship
between the "encoder" and "decoder", let's figure out how to train this sucker.
Recall Equation 9 shows us how define our objective in terms of distributions:
:math:`z, z|X, X|z`.  When we're optimizing, we don't directly work with
distributions, instead we have samples and a single objective function.
We can translate Equation 9 into this goal by taking the expectation over
the relevant variables like so:

.. math::

    E_{X\sim D}[\log P(X) - \mathcal{D}(Q(z|X) || P(z|X))] = 
        E_{X\sim D}[E_{z\sim Q}[\log P(X|z)] - \mathcal{D}(Q(z|X) || P(z))]  \\
    \tag{12}

To be a bit pedantic, we still don't have a function we can't optimize directly
because we don't know how to take the expectation.  Of course, we'll just
make the usual approximations by replacing the expectation with summations. 
I'll show it step by step, starting from the RHS of Equation 12:

.. math::

    &E_{X\sim D}[E_{z\sim Q}[\log P(X|z)] - \mathcal{D}(Q(z|X) || P(z))] \\
    &= E_{X\sim D}[E_{\epsilon \sim \mathcal{N}(0, I)}[
        \log P(X|z=\mu_{z|X}(X) + \Sigma_{z|X}^{1/2}(X)*\epsilon)
    ] - \mathcal{D}(Q(z|X) || P(z))]  \\
    &\approx \frac{1}{N}\sum_{x_i \in X}
       E_{\epsilon \sim \mathcal{N}(0, I)}[
        \log P(x_i|z=\mu_{z|X}(x_i) + \Sigma_{z|X}^{1/2}(x_i)*\epsilon)
       ] - \mathcal{D}(Q(z|x_i) || P(z))  \\
    &\approx \frac{1}{N}\sum_{x_i \in X}
        \log P(x_i|z=\mu_{z|X}(x_i) + \Sigma_{z|X}^{1/2}(x_i)*\epsilon)
       - \mathcal{D}(Q(z|x_i) || P(z))  \\

    \tag{13}

Going line by line: first, we use our "reparameterization trick" by just
sampling from our isotropic normal distribution instead of our prior :math:`z`.
Next, let's approximate the outer expectation by taking our N observations of
the :math:`X` values and averaging them.  This is what we implicitly do in most
learning algorithms when we assume i.i.d. Finally, we'll simplify the inner
expectation by just using a single sample paired with each observation
:math:`x_i`.  This requires a bit more explanation.

We want to simplify the inner expectation :math:`E_{\epsilon \sim
\mathcal{N}(0, I)}[\cdot]`.  To compute this, each time we evaluate the network, we
must explicitly sample a *new* value of :math:`\epsilon` from our isotropic
normal distribution.  That means we can just pair each observation
:math:`x_i` with a bunch of samples from :math:`\mathcal{N}(0, I)`
to make a "full input".

However, instead of doing that explicitly, let's just pair each :math:`x_i`
with a single sample from :math:`\mathcal{N}(0, I)`.  If we're training over
many epochs (large loop over all :math:`x_i` values), it's as if we are pairing
each observation with a bunch of new values sampled from :math:`\mathcal{N}(0,
I)`.
According to stochastic gradient descent theory, these two methods should
converge to the same place and it simplifies life for us a bit.

As a final note, we can simplify the last line in Equation 13 to something
like:

.. math::

    \frac{1}{N} \sum_{x_i \in X} -\frac{1}{2\sigma^2}(x_i-\mu_{X|z})^2 - \frac{1}{2}\big(tr(\Sigma_{z|X}(x_i)) + (\mu_{z|X}(x_i))^T(\mu_{z|X}(x_i)) - k - \log \text{det}(\Sigma_{z|X}(x_i))\big) \tag{14}

Each of the two big terms corresponds to :math:`\log P(x_i|z)` and 
:math:`\mathcal{D}(Q(z|x_i) || P(z)` respectively, where I dropped out some of
the constants from :math:`\log P(x_i|z)` (since they're not needed in the gradient) 
and used the formula for `KL divergence between two multivariate normals
<https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Kullback.E2.80.93Leibler_divergence_for_multivariate_normal_distributions>`__.

And now that we have a concrete formula for the overall objective, it's straight 
forward to perform stochastic gradient descent either by manually working the
derivatives or using some form of automatic differentiation.

|h3| 3.4 Generating New Samples |h3e|

Finally, we get to the interesting part of our model.  After all that training
we can finally try to generate some new observations using our implicit
generative model.  Even though we did all that work with the
"reparameterization trick" and KL divergence, we still only need our implicit
generative model from Section 2.1.

.. figure:: /images/variational_autoencoder4.png
  :width: 250px
  :alt: Variational "Decoder" Diagram
  :align: center

  Figure 5: The generative model component of a variational autoencoder
  in test mode.

Figure 5 shows our generative model.  To generate a new observation, all
we have to do is sample from our isotropic normal distribution (our prior
for our latent variables), and then run it through our neural network.
The network should have learned how to transform our latent variables into
the mean of what our training data looks like [2]_.

Note, that our network now only outputs the mean of our generative output,
we can additionally sample from our actual output distribution if we sample a
normal distribution with this mean.  In most cases, the mean is probably what
we want though (e.g. when generating an image, the mean values are just the
values for the pixels).

|h3| 4. A Basic Example: MNIST Variational Autoencoder |h3e|

The nice thing about many of these modern ML techniques is that implementations are
widely available.  I put together a 
`notebook <https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational-autoencoder.ipynb>`__ 
that uses `Keras <https://keras.io/>`__ to build a variational autoencoder [3]_.
The code is from the Keras convolutional variational autoencoder example and
I just made some small changes to the parameters.  I also added some annotations
that make reference to the things we discussed in this post.

Figure 6 shows a sample of the digits I was able to generate with 64 latent
variables in the above Keras example.

.. figure:: /images/generated_digits.png
    :height: 350px
    :alt: MNIST digits generated from a variational autoencoder model
    :align: center

    Figure 6: MNIST digits generated from a variational autoencoder model.

Not the prettiest hand writing =) We definitely got some decent looking digits
but also some really weird ones.  Usually the explanation for the weird ones
are that they're in between two digits or two styles of writing the same digit
(or maybe I didn't train the network well?).
An example might be the top left digit.  Is it a "3", "5" or "6"?
Kind of hard to tell.

Anyways take a look at the 
`notebook <https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational-autoencoder.ipynb>`__,
I find it really interesting to see the connection between theory and
implementation.  It's often that you see an implementation and it's very
difficult to reverse-engineer it back to the theory because of all the
simplifications that have been done.  Hopefully this post and the accompanying
notebook will help.

|h2| Conclusion |h2e|

Variational autoencoders are such a cool idea: it's a full blown probabilistic
latent variable model which you don't need explicitly specify!  On top of that,
it builds on top of modern machine learning techniques, meaning that it's also
quite scalable to large datasets (if you have a GPU).  I'm a big fan of
probabilistic models but an even bigger fan of practical things, which is why
I'm so enamoured with the idea of these variational techniques in ML.  I plan
on continuing in this direction of exploring more of these techniques in ML
that have a solid basis in probability.  Look out for future posts!


|h2| Further Reading |h2e|

* Previous Posts: 
  `Variational Bayes and The Mean-Field Approximation <link://slug/variational-bayes-and-the-mean-field-approximation>`__, `Variational Calculus <link://slug/the-calculus-of-variations>`__
* Wikipedia: `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Generative Models <https://en.wikipedia.org/wiki/Generative_model>`__, `Autoencoders <https://en.wikipedia.org/wiki/Autoencoder>`__, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__, `Inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`__
* "Tutorial on Variational Autoencoders", Carl Doersch, https://arxiv.org/abs/1606.05908
  
|br|

.. [1] Said another way, the posterior distribution for each sample :math:`p(z|X=x_i)` has some distinctive shape that is probably very different from the prior :math:`p(z)`.  If we sample any given :math:`z_m\sim Z` (which has distribution :math:`p(z)`), then most of those samples will have low :math:`p(Z=z_m|X=x_i)` because the distributions have vastly different shapes.  This implies that, :math:`p(X=x_i|Z=z_m)` (recall Bayes theorem) will also be low, which implies little contribution to our likelihood in Equation 7.

.. [2] If you find this a bit confusing, here's another explanation.  The only reason we did all the work on the "encoder" part was to generate a good distribution for :math:`z|X`.  That is given an :math:`x_i`, find the likely :math:`z` values.  However, we made sure that when we average over all the :math:`X` observations, our average :math:`z` values would still match our prior :math:`p(z)` isotropic normal distribution via the KL divergence.  That means, sampling for our isotropic normal distribution should still give us likely values for :math:`z`.

.. [3] I initially just tried to use this example with just my CPU but it was painfully slow (~ 5+ min/epoch).  So I embarked on a multi-week journey to buy a modern GPU, re-build my computer and dual-boot Linux (vs. using a virtual machine).  The speed-up was quite dramatic, now it's around ~15 secs/epoch.


