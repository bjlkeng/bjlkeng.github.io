.. title: Bayesian Learning via Stochastic Gradient Langevin Dynamics and Bayes by Backprop
.. slug: bayesian-learning-via-stochastic-gradient-langevin-dynamics-and-bayes-by-backprop
.. date: 2022-11-23 21:25:40 UTC-05:00
.. tags: Bayesian, Bayes by Backprop, SGLD, variational inference, elbo, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

After a long digression, I'm finally back to one of the main lines of research
that I wanted to write about.  The two main ideas in this post are not that
recent but have been quite impactful (one of the 
`papers <https://icml.cc/virtual/2021/test-of-time/11808>`__ won a recent ICML
test of time award).  They address two of the topics that are near and dear to
my heart: Bayesian learning and scalability.  Dare I even ask who wouldn't be
interested in the intersection of these topics?

This post is about two techniques to perform scalable Bayesian inference.  They
both address the problem using stochastic gradient descent (SGD) but in very
different ways.  One leverages the observation that SGD plus some noise will
converge to Bayesian posterior sampling [Welling2011]_, while the other generalizes the
"reparameterization trick" from variational autoencoders to enable non-Gaussian
posterior approximations [Blundell2015]_.  Both are easily implemented in the modern deep
learning toolkit thus benefit from the massive scalability of that toolchain.
As usual, I will go over the necessary background (or refer you to my previous
posts), intuition, some math, and a couple of toy examples that I implemented.


.. TEASER_END
.. section-numbering::
.. raw:: html

    <div class="card card-body bg-light">
    <h1>Table of Contents</h1>

.. contents:: 
    :depth: 2
    :local:

.. raw:: html

    </div>
    <p>


Motivation
==========

Bayesian learning is all about learning the `posterior <https://en.wikipedia.org/wiki/Posterior_probability>`__ 
distribution for the parameters of your statistical model, which in turns allows
you to quantify the uncertainty about them.  The classic place to start is with
Bayes theorem:

.. math::

   p({\bf \theta}|{\bf x}) &= \frac{p({\bf x}|{\bf \theta})p({\bf \theta})}{p({\bf x})} \\
                           &= \text{const}\cdot p({\bf x}|{\bf \theta})p({\bf \theta}) \\
                           &= \text{const}\cdot \text{likelihood} \cdot \text{prior} \\
                           \tag{1}

where :math:`{\bf x}` is a vector of data points (often 
`IID <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`__)
and :math:`{\bf \theta}` is the vector of statistical parameters of your model.

Generally there is no closed form solution and you have to resort to heavy methods such
as Markov Chain Monte Carlo (MCMC) methods or some form of approximation (which
we'll get to later).  MCMC methods never give you a closed form but instead give
you samples from the posterior distribution, which you can use then use
to compute any statistic you like.  These methods are quite slow because they
rely on `Monte Carlo <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__
methods.

This brings us to our first scalability problem Bayesian learning: it does not
scale well with the number of parameters.  Randomly sampling with MCMC implies
that you have to "explore" the parameter space, which potentially grows
exponentially with the number of parameters.  There are many techniques to make
this more efficient but ultimately it's hard to compensate for an exponential.
The natural example for this situation is neural networks which can have orders
of magnitude more parameters compared to classic Bayesian learning problems
(I'll also add that the use-case of the posterior is usually different too).

The other non-obvious scalability issue with MCMC from Equation 1 is the data.
Each evaluation of MCMC requires an evaluation of the likelihood and prior from
Equation 1.  For large data (e.g. modern deep learning datasets), you quickly
hit issues either with memory and/or computation speed.

For non-Bayesian learning, modern deep learning has really solved both of these
problems by leveraging one of the simplest optimization method out there,
stochastic gradient descent, along with the massive compute power of modern
hardware (and its associated toolchain).  How can we leverage these
developments to scale Bayesian learning?  Keep reading to found out!

Background
==========

Bayesian Networks and Bayesian Hierarchical Models
--------------------------------------------------

We can take the idea of parameters and priors from Equation 1 to multiple
levels.  Equation 1 implicitly assumes that there is one "level" of parameters
(:math:`\theta`) that we're trying to estimate with prior distributions
(:math:`p({\bf \theta})`) attached to them, but there's no reason why you only
need a single level.  In fact, our parameters can be conditioned on parameters,
which can be conditioned on parameters, and so on.  
This is called `Bayesian hierarchical modeling <https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling>`__.
If this sounds oddly familiar, it's the same thing as `Bayesian networks
<https://en.wikipedia.org/wiki/Bayesian_network#Graphical_model>`__ in a different context (if you're
familiar with that).  My `previous post
<link://slug/the-expectation-maximization-algorithm>`__ gives a high level
summary on the intuition with latent variables.

To quickly summarize, in a parameterized statistical model there are broadly
two types of variables: observed and unobserved.  Observed variables are ones
where we have values and often assumed to be
`IID <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`__.

Unobserved variables are ones we don't have values for and go by several names.
In Bayesian networks they
are usually called latent or hidden (random) variables, which can have 
complex conditional dependencies specified as a DAG.  In hierarchical models
they are called **hyperparameters**, which are the parameters of the 
observed models, the parameters of parameters, parameters of parameters of
parameters and so on.  Similarly, each of these hyperparameters has a 
distribution which we call a **hyperprior**.  

These two concepts are mathematically the same and (from what I gather) the
difference is really just their interpretation.  In the context of hierarchical models,
the hyperparameters and hyperpriors represent some structural knowledge
about the problem, hence of the use of term "priors".  The data is typically
believed to appear in hierarchical "clusters" that share similar attributes
(i.e., drawn from the same distribution).  This view is more typical in
Bayesian statistics applications where the number of stages (and thus
variables) is usually small (two or three).  If terms such as 
`fixed or random effects models <https://en.wikipedia.org/wiki/Multilevel_model>`__
ring a bell then this framing will make much more sense.

In Bayesian networks, the latent variables can represent the underlying
phenomenon but also can be artificially introduced to make the problem more
tractable.  This happens more often in machine learning such as in `variational
autoencoders <link://slug/variational-autoencoders>`__.  In these contexts,
they are often modeling a much bigger network and can have arbitrarily large
stages and network sizes.  By varying assumptions on the latent variables and
their connectivity, there are many efficient algorithms that can perform either
approximate or exact inference on them.  Most applications in ML seem to follow
the Bayesian networks nomenclature since its context is more general.  We'll
stick with this framing since most of the ML sources will explain it this way.


Markov Chain Monte Carlo and Hamiltonian Monte Carlo
----------------------------------------------------

This subsection gives a brief introduction Monte Carlo Markov Chains (MCMC) and
Hamiltonian Monte Carlo.  I've written about both
`here <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__ 
and `here <link://slug/hamiltonian-monte-carlo>`__ if you want the nitty gritty details
(and better intuition).

`MCMC <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`__ methods are a
class of algorithm for sampling from a target probability distribution 
(e.g., posterior distribution).  The most basic algorithm is relatively simple,
starting from a given point:

1. Propose a new point (state)
2. With some probability calculated using the target distribution (or some
   function proportional to it), transition and accept this new point (state); 
   otherwise, stay at the current point (state).
3. Repeat steps 1 and 2, and periodically output the current point (state).

Many MCMC algorithms follow this general framework.  The key is ensuring
that the proposal and the acceptance probability define a Markov chain such
that its stationary distribution (i.e., steady state) is the same as your
target distribution.  See my previous post on `MCMC <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__ for more details.

Two additional complications.  The first complication is that your initial
state may be in some weird region that causes the algorithm to explore parts of
the state space that are low probability.  To solve this, you can perform
"burn-in" by starting the algorithm and throwing away a bunch of 
states to have a higher change to be in a more "normal" region of the state
space.  The other complication is that sequential samples will often be correlated,
but you almost always want independent samples.  Thus (as specified in the steps
above), we only periodically output the current state as a sample to ensure
that the we have minimal correlation.  This is generally called "thinning".  A
well tuned MCMC algorithm will have both a high acceptance rate and little
correlation between samples.

`Hamiltonian Monte Carlo <https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo>`__  (HMC)
is a popular MCMC algorithm that has a high acceptance rate with low
correlation between samples.  At a high level, it transforms the sampling of a target probability
distribution into a physics problem with `Hamiltonian dynamics <https://en.wikipedia.org/wiki/Hamiltonian_mechanics>`__.
Intuitively, the problem is similar to a frictionless puck moving along a surface representing
our target distribution.
The position variables :math:`q` represent the state from our probability
distribution, and the momentum :math:`p` (equivalently velocity) are a set of
instrument variables to make the problem work.  For each proposal point, we
randomly pick a new momentum (and thus energy level of the system) and simulate
from our current point.  The end point is our new proposal point.

This is effectively simulating the associated differential equations of this
physical system.  It works well because the produced proposal point has both a high
acceptance rate and can easily be "far away" with more simulation steps (thus
low correlation).  In fact, the acceptance rate would be 100% if it not for the
fact that we have some discretization error from simulating the differential
equations.  See my previous post on `HMC <link://slug/hamiltonian-monte-carlo>`__ for more details.

A common method for simulation of this physics problem uses the "leap frog" method
where we discretize time and simulate time step-by-step:

.. math::

   p_i(t+\epsilon/2) &= p_i(t) - \frac{\epsilon}{2} \frac{\partial H}{\partial q_i}(q(t)) \tag{2}\\
   q_i(t+\epsilon) &= q_i(t) + \epsilon \frac{\partial H}{\partial p_i}(p(t+\epsilon/2)) \tag{3} \\
   p_i(t+\epsilon) &= p_i(t+\epsilon/2) - \frac{\epsilon}{2} \frac{\partial H}{\partial q_i}(q(t+\epsilon)) \tag{4}

Where :math:`i` is the dimension index, :math:`q(t)` represent the position
variables at time :math:`t`, :math:`p(t)` similarly represent the momentum
variables, :math:`\epsilon` is the step size of the discretized simulation, and
:math:`H := U(q) + K(p)` is the Hamiltonian, which (in this case) equals the
sum of potential energy :math:`U(q)` and the kinetic energy :math:`K(p)`.  The
potential energy is typically the negative logarithm of the target density up
to a constant :math:`f({\bf q})`, and the kinetic energy is usually defined as
independent zero-mean Gaussians with variances :math:`m_i`:

.. math::

   U({\bf q}) &= -log[f({\bf q})]  \\
   K({\bf p}) &= \sum_{i=1}^D \frac{p_i^2}{2m_i}  \\
   \tag{5}

Once we have a new proposal state :math:`(q^*, p^*)`, we accept the new state
according to this probability using a 
`Metropolis-Hasting <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`__ update:

.. math::

       A(q^*, p^*) = \min[1, \exp\big(-U(q^*) + U(q) -K(p^*)+K(p)\big)] \tag{6}

Langevin Monte Carlo
--------------------

Langevin Monte Carlo (LMC) [Radford2012]_ is a special case of HMC where we only
take a *single* step in the simulation to propose a new state (versus multiple
steps in a typical HMC algorithm).  It is sometimes referred to as the
Metropolis-Adjusted-Langevin algorithm (MALA) (see [Teh2015]_ for more
details).  With some simplification, we will see that a new familiar
behavior emerges from this special case.

Suppose we define kinetic energy as :math:`K(p) = \frac{1}{2}\sum p_i^2`,
which is typical for a HMC formulation.  Next, we set our momentum :math:`p` as
a sample from a zero mean, unit variance Gaussian (still same as HMC). 
Finally, we run a single step of the leap frog to get new a new proposal state 
:math:`q^*` and :math:`p^*`.

We only need to focus on the position :math:`q` because we resample the
momentum :math:`p` on each new proposal state, which gets reset each step
anyways.  Starting from Equation 3:

.. math::

   q_i^* &= q_i(t) + \epsilon \frac{\partial H}{\partial p}(p(t+\epsilon/2))  \\
       &= q_i(t) + \epsilon \frac{\partial [U(q) + K(p)]}{\partial p}(p(t+\epsilon/2))  \\
       &= q_i(t) + \epsilon \frac{\partial [U(q) + \frac{1}{2}\sum p_i^2]}{\partial p}(p(t+\epsilon/2))  && \text{Per def. of kinetic energy} \\
       &= q_i(t) + \epsilon p|_{p=p(t+\epsilon/2)}  \\
       &= q_i(t) + \epsilon [p(t) - \frac{\epsilon}{2} \frac{\partial H}{\partial q_i}(q(t))] && \text{Eq. } 2 \\
       &= q_i(t) - \frac{\epsilon^2}{2} \frac{\partial H}{\partial q_i}(q(t)) + \epsilon p(t) \\
       &= q_i(t) - \frac{\epsilon^2}{2} \frac{\partial U}{\partial q_i}(q(t)) + \epsilon p(t) && H := U(q) + K(p) \\
   \tag{7}

Equation 7 is known in physics as (one type of) Langevin Equation (see box for explanation),
thus the name Langevin Monte Carlo.

Now that we have a proposal state (:math:`q^*`), we can view the algorithm
as running a vanilla Metropolis-Hastings update where the proposal is coming
from a Gaussian with mean :math:`q_i(t) - \frac{\epsilon^2}{2} \frac{\partial U}{\partial q_i}(q(t))`
and variance :math:`\epsilon^2` corresponding to Equation 7.
By eliminating :math:`p` (and the associated :math:`p^*`, not shown here) from
the original HMC acceptance probability in Equation 6, we can derive the
following expression:

.. math::

   A(q^*) = \min\big[1, \frac{\exp(-U(q^*))}{\exp(-U(q))} 
        \Pi_{i=1}^d 
            \frac{\exp(-(q_i - q_i^* + (\epsilon^2 / 2) [\frac{\partial U}{\partial q_i}](q^*))^2 / 2\epsilon^2)}
            {\exp(-(q_i^* - q_i + (\epsilon^2 / 2) [\frac{\partial U}{\partial q_i}](q))^2 / 2\epsilon^2)}\big] \\
    \tag{8}

Even though LMC is derived from HMC, its properties are quite different.
The movement between states will be a combination of the :math:`\frac{\epsilon^2}{2} \frac{\partial U}{\partial q_i}(q(t))`
term and the :math:`\epsilon p(t)`.  Since :math:`\epsilon` is necessarily
small (otherwise your simulation will not be accurate), the former term
will be very small and the latter term will resemble a simple
Metropolis-Hastings random walk.  A big difference though is that LMC
has better scaling properties when increasing dimensions.  See [Radford2012]_
for more details.

Finally, we'll want to re-write Equation 7 using different notation
to line up with our usual notation for stochastic gradient descent.
First, we'll use :math:`\theta` instead of :math:`q` to imply that
we're sampling from parameters of our model.  Next, we'll
rewrite the potential energy :math:`U(\theta)` as the likelihood times prior
(where :math:`x_i` are our observed data points):

.. math::

    U(\theta_t) &= -log[f(\theta_t)] \\
                &= -\log[p(\theta_t)] - \sum_{i=1}^N \log[p(x_i | \theta_t)] \\
    \tag{9}

Simplifying our Equation 7, we get:

.. math::

    
    \theta_{t+1} &= \theta_t - \frac{\epsilon_0^2}{2} \frac{\partial U(\theta)}{\partial \theta} + \epsilon_0 p(t) \\
    \theta_{t+1} &= \theta_t- \frac{\epsilon_0^2}{2} \frac{\partial [-\log[p(\theta_t)] - \sum_{i=1}^N \log[p(x_i | \theta_t)]]}{\partial \theta} + \epsilon_0 p(t) && \text{Eq. } 10\\
    \theta_{t+1} - \theta_t &= \frac{\epsilon_0^2}{2} \big (\nabla \log[p(\theta_t)] + \sum_{i=1}^N \nabla \log[p(x_i | \theta_t)]]\big) + \epsilon_0 p(t) \\
    \theta_{t+1} - \theta_t &= \frac{\epsilon}{2} \big (\nabla \log[p(\theta_t)] + \sum_{i=1}^N \nabla \log[p(x_i | \theta_t)]]\big) + \sqrt{\epsilon} p(t) && \epsilon := \epsilon_0^2\\
    \Delta \theta_t &= \frac{\epsilon}{2} \big (\nabla \log[p(\theta_t)] + \sum_{i=1}^N \nabla \log[p(x_i | \theta_t)]]\big) + \varepsilon && \varepsilon \sim N(0, \epsilon) \\
    \tag{10}

Which looks eerily like gradient descent except that we're adding Gaussian
noise at the end. Stay tuned!

.. admonition:: Langevin's Diffusion

   In the field of stochastic differential equations, a general Itô diffusion
   process is of the form:

   .. math::
    
       dX_t = a(X_t, t)dt + b(X_t, t)dW_t \tag{A.1}

   where :math:`X_t` is a stochastic process, :math:`W_t` is a Weiner process
   and :math:`a(\cdot), b(\cdot)` are functions of :math:`X_t, t`.  The form 
   of Equation A.1 is the differential form.  See my post on
   `Stochastic Calculus <link://slug/an-introduction-to-stochastic-calculus>`__ 
   for more details.

   One of the forms of Langevin diffusion is a special case of Equation A.1:

   .. math::
    
       dq_t &= -\frac{1}{2}\frac{dU(q_t)}{dq} dt + dW_t \\
            &= -\frac{1}{2}\nabla U(q_t) dt + dW_t \\
       \tag{A.2}

   Where :math:`q_t` is the position, :math:`U` is the potential energy,
   :math:`\frac{dU}{dq}` is the force (position derivative of potential
   energy), and :math:`W_t` is the Wiener process.  
  
   In the context of MCMC, we model the potential energy of this system as
   :math:`U(q) = \log f(q)` where :math:`f` is proportional to the likelihood
   times prior as is usually required in MCMC methods.  With this substition,
   Equation A.2 is the same as Equation 11 except a continuous time version of
   it.  To see this more clearly, it is important to note that the increments
   of the standard Weiner process :math:`W_t` are zero-mean Gaussians with
   variance equal to the time difference.  Once discretized with stepsize
   :math:`\epsilon`, this precisely equals our :math:`\varepsilon` sample from
   Equation 10.




Stochastic Gradient Descent and RMSprop
---------------------------------------

I'll only briefly cover stochastic gradient descent because I'm assuming most
readers will be very familiar with this algorithm.  
`Stochastic gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`__ (SGD)
is an iterative stochastic optimization of gradient descent.  The main difference
is that it uses a randomly selected subset of the data to estimate gradient at 
each step.  For a given statistical model with parameters :math:`\theta`,
log prior :math:`\log p(\theta)`, and log likelihood :math:`\sum_{i=1}^N \log[p(x_i | \theta_t)]]`
with observed data points :math:`x_i`, we have:

.. math::

    \Delta \theta_t = \frac{\epsilon_t}{2} \big (\nabla \log[p(\theta_t)] 
    + \frac{N}{n} \sum_{i=1}^n \nabla \log[p(x_{ti} | \theta_t)]]\big) 
      \tag{11}

where :math:`\epsilon_t` is a sequence of step sizes, and each iteration :math:`t`
we have a subset of :math:`n` data points called a *mini-batch*
:math:`X_t = \{x_{t_1}, \ldots, x_{t_n}\}`.
By using an approximate gradient over many iterations the entire dataset is
eventually used, and the noise in the estimated gradient averages out.
Additionally for large datasets where the estimated gradient is accurate
enough, this gives significant computational savings versus using the whole
dataset at each iteration.

Convergence to a local optimum is guaranteed with some mild assumptions combined
with a major requirement that the step size schedule :math:`\epsilon_t` satisfies:

.. math::

   \sum_{t=1}^\infty \epsilon_t = \infty \hspace{50pt} \sum_{t=1}^\infty \epsilon_t^2 < \infty
   \tag{12}

Intuitively, the first constraint ensures that we make progress to reaching the
local optimum, while the second constraint ensures we don't just bounce around
that optimum.  A typical schedule to ensure that this is the case is using
a decayed polynomial:

.. math::

   \epsilon_t = a(b+t)^{-\gamma} \tag{13}

with :math:`\gamma \in (0.5, 1]`.

One of the issues with using vanilla SGD is that the gradients of the model
parameters (i.e. dimensions) may have wildly different variances.  For example,
one parameter may be smoothly descending at a constant rate while another may be
bouncing around quite a bit (especially with mini-batches).  To solve this, many
variations on SGD have been proposed that adjust the algorithm to account for the
variation in parameter gradients.  

`RMSprop <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp>`__
is a popular variant that is conceptually quite simple.  It adjusts the
learning rate *per parameter* to ensure that all of the learning rates are roughly
the same magnitude.  It does this by keeping a running average of the magnitudes
of recent gradients for parameter :math:`\theta` as :math:`v(\theta, t)`.
For :math:`j^{th}` parameter :math:`\theta^j` in iteration :math:`t`, we have:

.. math::

   v(\theta^j, t) := \gamma v(\theta^j, t-1) + (1-\gamma)(\nabla Q_i(\theta^j))^2 \tag{14}

where :math:`Q_i` is the loss function, and :math:`\gamma` is the smoothing
constant of the average with typical value set at `0.99`.  With :math:`v(\theta^j, t)`,
the update becomes:

.. math::

   \Delta \theta^j := - \frac{\epsilon_t}{\sqrt{v(\theta^j, t)}} \nabla Q_i(\theta^j) \tag{15}

From Equation 15, when you have large gradients (:math:`\nabla Q >1`), it scales
the learning rate down; while if you have small gradients (:math:`\nabla Q < 1`),
it scales the learning rate up.  If :math:`\nabla Q` is constant in each
parameter but with different magnitudes, it will update each parameter by the
learning rate :math:`\epsilon_t`, attempting to descend each dimension at the same
rate.  Empirically, these variations of SGD are necessary to make SGD practical
for a wide range of models.

Variational Inference and the Reparameterization Trick
------------------------------------------------------

I've written a lot about variational inference in past posts so I'll
keep this section brief and only touch upon the relevant parts.
If you want more detail and intuition, check out my posts on 
`Semi-supervised learning with Variational Autoencoders <link://slug/semi-supervised-learning-with-variational-autoencoders>`__,
and `Variational Bayes and The Mean-Field Approximation <link://slug/variational-bayes-and-the-mean-field-approximation>`__.

As we discussed above, our goal is to find the posterior, :math:`p(\theta|X)`,
that tells us the distribution of the :math:`\theta` parameters.  Unfortunately,
this problem is intractable for all but the simplest problems. How can we 
overcome this problem? Approximation! 

We'll approximate :math:`p(\theta|X)` by another known distribution :math:`q(\theta|\phi)` 
parameterized by :math:`\phi`.  Importantly, :math:`q(\theta|\phi)` often has
simplifying assumptions about its relationships with other variables. 
For example, you might assume that they are all independent of each other
e.g., :math:`q(\theta|\phi) = \pi_{i=1}^n q_i(\theta_i|\phi_i)` (a example of mean-field approximation).

The nice thing about this approximation is that we turned our intractable Bayesian learning problem
into an optimization one where we just want to find the parameters :math:`\phi`
of :math:`q(\theta|\phi)` that best match our posterior :math:`p(\theta|X)`.
How well our approximation matches our posterior is both dependent on the
functional form of :math:`q` as well as our optimization procedure.

In terms of "best match", the standard way of measuring it is to use
`KL divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__.
Without going into the derivation 
(see my `previous post <link://slug/semi-supervised-learning-with-variational-autoencoders>`__),
if we start from the KL divergence between our approximate posterior and exact posterior,
we'll arrive at the evidence lower bound (ELBO) for a single data point
:math:`X`:

.. math::

  D_{KL}(Q||P) &= E_q\big[\log \frac{q(\theta|\phi)}{p(\theta,X)}\big] + \log p(X) \\
  \log{p(X)} &\geq -E_q\big[\log\frac{q(\theta|\phi)}{p(\theta,X)}\big]  \\
             &= E_q\big[\log p(\theta,X) - \log q(\theta|\phi)\big] \\
             &= E_q\big[\log p(X|\theta) + \log p(\theta) - \log q(\theta|\phi)\big] \\
             &= E_q\big[\text{likelihood} + \text{prior} - \text{approx. posterior} \big] \\
              \tag{16}

The left hand side of Equation 16 is constant (with respect to the observed
data), so maximizing the right hand side achieves our desired goal.  It just so
happens this looks a lot like finding a 
`MAP <https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation>`__ with a
likelihood and prior term.  The two differences are that (a) we have an additional term
for our approximate posterior and (b) we have to take the expectation with respect
to samples from that approximate posterior.  When using a SGD approach, we can
sample points from the :math:`q` distribution and use it to approximate the
expectation in Equation 16.  In many cases though, it's not obvious how to
sample from :math:`q` because you also need to backprop through it.  

In the case of 
`variational autoencoders <link://slug/variational-autoencoders>`__,
we define an approximate Gaussian posterior :math:`q(z|\phi)` on the latent variables
:math:`z`. This approximate posterior is defined by a neural network with
weights :math:`\phi` that output a mean and variance representing the
parameters of the Gaussian.  We will want to sample from :math:`q` to
approximate the expectation in Equation 16, but also backprop through :math:`q`
to update the weights :math:`\phi` of the approximate posterior.
You can't directly backprop through it but you can reparameterize it by
using a standard normal distribution, starting from Equation 16 (using
:math:`z` instead of :math:`\theta`):

.. math::

        &E_{z\sim q}\big[\log p(X|z) + \log p(z) - \log q(z|\phi)\big] \\
        &= E_{\epsilon \sim \mathcal{N}(0, I)}\big[(\log p(X|z) + \log p(z) - \log q(z|\phi))\big|_{z=\mu_z(X) + \Sigma_z^{1/2}(X) * \epsilon}\big] \\
        &\approx (\log p(X|z) + \log p(z) - \log q(z|\phi))\big|_{z=\mu_z(X) + \Sigma_z^{1/2}(X) * \epsilon} \\
        \tag{17}

where :math:`\mu_z` and :math:`\Sigma_z` are the mean and covariance matrix of
the approximate posterior, and :math:`\epsilon` is a sample from a standard Gaussian.
This is commonly referred to as the "reparameterization trick" where instead of
directly computing :math:`q` you just scale and shift a standard normal
distribution.  Thus, you can still backprop through the mean and covariances.
The last line approximates the expectation by taking a single sample, which
often works fine when using SGD.

Stochastic Gradient Langevin Dynamics 
=====================================

Stochastic Gradient Langevin Dynamics (SGLD) combines the ideas of Langevin
Monte Carlo (Equation 10) with Stochastic Gradient Descent (Equation 11)
given by:

.. math::

    \Delta \theta_t &= \frac{\epsilon_t}{2} \big (\nabla \log[p(\theta_t)] + \frac{N}{n} \sum_{i=1}^n \nabla \log[p(x_{ti} | \theta_t)]\big) + \varepsilon \\
    \varepsilon &\sim N(0, \epsilon_t)  \\
    \tag{18}

This results in an algorithm that is mechanically equivalent to SGD except with
some Gaussian noise added to each parameter update.  Importantly though, there
are several key choices that SGLD makes:

* :math:`\epsilon_t` decreases towards zero just as in SGD.
* Balance the Gaussian noise :math:`\varepsilon` variance with the step size
  :math:`\epsilon_t` as in LMC.
* Ignore the Metropolis-Hastings updates (Equation 8) using the fact that
  rejection rates asymptotically go to zero as :math:`\epsilon_t \to 0`. 

This algorithm has the advantage of SGD of being able to work on large data
sets (because of the mini-batches) while still computing uncertainty
(using LMC-like estimates).  The avoidance of the Metropolis-Hastings update is
key so that an expensive evaluation of the whole dataset is not needed at each
iteration.

The intuition here is that in earlier iterations this will behave much like SGD
stepping towards a local minimum because the large gradient overcomes the
noise.  In later iterations with a small :math:`\epsilon_t`, the noise
dominates and the gradient plays a much smaller role resulting in each
iteration bouncing around the local maxima via a random walk (with a bias
towards the local minimum from the gradient).  Additionally, in between these two
extremes, the algorithm should vary smoothly.  Thus with carefully selected
hyperparameters, you can *effectively* sample from the posterior distribution
(more on this later).

What is not obvious though is that why this should give correct the correct
result.  It surely will be able to get close to a local minimum (similar to
SGD) but why would it give the correct uncertainty estimates without the
Metropolis-Hastings update step?  This is the topic of the next subsection.

Correctness of SGLD 
-------------------

*Note:* [Teh2015]_ *has the hardcore proof of SGLD correctness versus a very
informal sketch presented in the original paper* ([Welling2011]_) *.  I'll mainly
stick to the original paper's presentation (mostly because the hardcore proof
is way beyond my comprehension), but will call out a couple of notable things.*

To set up this problem, let us first define several quantities.
First define the true gradient of the log probability,
which is just the negative gradient our usual `MAP <https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation>`__
loss function (with no mini-batches):

.. math::

   g(\theta) = \nabla \log p(\theta) + \sum_{i=1}^N \nabla \log p(X_i|\theta) \tag{19}

Next, let's define another related quantity:

.. math::

   h_t(\theta) = \nabla \log p(\theta) + \frac{N}{n}\sum_{i=1}^n \nabla \log p(X_{ti}|\theta) - g(\theta) \tag{20}

Equation 20 is essentially the difference between our SGD update (with
mini-batch :math:`t`) and the true gradient update (with all the data).
Notice that :math:`h_t(\theta) + g(\theta)` is just an SGD update 
which can be obtained by canceling out the last term.

Importantly, :math:`h_t(\theta)` is a zero-mean random variable with
finite variance :math:`V(\theta)`.  Zero-mean because we're subtracting out the
true gradient so our random mini-batches should not have any bias.  Similarly,
the randomness comes from the fact that we're randomly selecting finite
mini-batches, which should yield only a finite variance.

With these quantities, we can rewrite Equation 18 using the fact
that :math:`h_t(\theta) + g(\theta)` is an SGD update:

.. math::

    \Delta \theta_t &= \frac{\epsilon_t}{2} \big (g(\theta_t) + h_t(\theta_t) \big) + \varepsilon \\
    \varepsilon &\sim N(0, \epsilon_t)  \\
    \tag{21}

With the above setup, we'll show two statements:

1. **Transition**: When we have large :math:`t`, the state transition
   of Equation 18/21 will be the same as LMC, that is, have its equilibrium
   distribution be the posterior distribution.
2. **Convergence**: There exists a subsequence of :math:`\theta_1,
   \theta_2, \ldots` that converges to the posterior distribution.

With these two shown, we can see that SGLD (for large :math:`t`) will
eventually get into a state where we can *theoretically* sample the posterior
distribution by taking the appropriate subsequence.  The paper makes a stronger
argument that the subsequence convergence implies convergence of the entire
sequence but it's not clear to me that it is the case.  At the end of this
subsection, I'll also mention a theorem from the rigorous proof ([Teh2015]_)
that gives a practical result where this may not matter.

**Transition**

We'll argue that Equation 18/21 converges to the same transition probability
as LMC and thus its equilibrium distribution will be the posterior.

First notice that Equation 18/21 is the same equation as LMC (Equation 10) except for the
additional randomness due to the mini-batches: :math:`\frac{N}{n} \sum_{i=1}^n \nabla \log[p(x_{ti} | \theta_t)]`.
This term is multiplied by a :math:`\frac{\epsilon_t}{2}` factor whereas
the standard deviation from the :math:`\varepsilon` term is :math:`\sqrt{\epsilon_t}`.
Thus as :math:`\epsilon_t \to 0`, the error from the mini-batch term will
vanish faster than the :math:`\varepsilon` term, converging to the LMC proposal
distribution (Equation 10).  That is, at large :math:`t` it approximates LMC
and eventually converges to it in the limit since the gradient update (and the
difference between the two) vanishes.

Next, we observe that LMC is a special case of HMC.  HMC is actually a
discretization of a continuous time differential equation.  The discretization
introduces error in the calcluation, which is the only reason why we need a
Metropolis-Hastings update (see previous post on `HMC <link://slug/hamiltonian-monte-carlo>`__).
However as :math:`\epsilon_t \to 0`, this error becomes negligible converging
to the continuous time dynamics, implying a 100% acceptance rate.
Thus, there is no need for an MH update for very small :math:`\epsilon_t`. 

In summary for large :math:`t`, the :math:`t^{th}` iteration of Equation
18/21 closely approximates the LMC Markov chain transition with very small error
so its equilibrium distribution closely approximates the desired posterior.
This would be great if we had a fixed :math:`t` but we are actually shrinking
:math:`t` towards 0 (as is needed by SGD), thus SGLD actually defines a
non-stationary Markov Chain and so we still need to show the actual sequence
will convert to the posterior.

**Convergence**

We will show that there exists some sequence of samples :math:`\theta_{t=a_1},
\theta_{t=a_2}, \ldots` that converge to the posterior for some strictly
increasing sequence :math:`a_1, a_2, \ldots` (note: the sequence is not
sequential e.g., :math:`a_{n+1}` is likely much bigger than :math:`a_{n+1}`).

First we fix a small :math:`\epsilon_0` such that :math:`0 < \epsilon_0 << 1`.
Assuming :math:`\{\epsilon_t\}` satisfy the decayed polynomial property from
Equation 13, there exists an increasing subsequence :math:`\{a_n \}` such that 
:math:`\sum_{t=a_n+1}^{a_{n+1}} \epsilon_t \to \epsilon_0` as :math:`n \to \infty`
(note: the :math:`+1` in the sum's upper limit is in the subscript, while the
lower limit is not).
That is, we can split the sequence :math:`\{\epsilon_t\}` into non-overlapping
segments such that successive segment approaches :math:`\epsilon_0`.  This can
be easily constructed by continually extending the current run until you go
over :math:`\epsilon_0`.  Since :math:`\epsilon_t` is decreasing, and we are
guaranteed that the sequence doesn't converge (Equation 12), we can always
construct the next segment with a smaller error that the previous one.

For large :math:`n`, if we look at each segment, the total Gaussian noise
injected will be the sum of each of the Gaussian noise injections.  The
`variance of sums of independent Gaussians <https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables>`__ 
is just the sum of the variances, so the total variance will be 
:math:`O(\epsilon_0)`.  Thus, the injected noise (standard deviation)
will be on the order of :math:`O(\sqrt{\epsilon})`.  Given this,
next we will want to show that the variance from the mini-batch error is
dominated by this injected noise.

To start, since :math:`\epsilon_0 << 1`, we have 
:math:`||\theta_t-\theta_{t=a_n}|| << 1` for :math:`t \in (a_n, a_{n+1}]` 
since the updates from Equation 18/21 cannot stray too far from where it
started.  Assuming the gradients vary smoothly (a key assumption) then
we can see the total update without the injected noise for a segment 
:math:`t \in (a_n, a_{n+1}]` is (i.e., Equation 21 minus the noise :math:`\varepsilon`):

.. math::

   \sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2}\big(g(\theta_t) + h_t(\theta_t)\big)
   = \frac{\epsilon_0}{2} g(\theta_{t=a_n}) + O(\epsilon_0) + \sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2} h_t(\theta_t) \tag{22}

We see that the :math:`g(\cdot)` summation expands into the gradient at
:math:`\theta_{t=a_n}` plus an error term :math:`O(\epsilon_0)`.  This is
from our assumption of :math:`||\theta_t-\theta_{t=a_n}|| << 1` plus
the gradients varying smoothly (`Lipschitz contiuity <https://en.wikipedia.org/wiki/Lipschitz_continuity>`__),
which imply that the difference between successive gradients will also be much
smaller than 1 (for an appropriately small :math:`\epsilon_0`).  Thus, the
error from this term on this segment will
be :math:`\sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2} O(1) = O(\epsilon_0)` as
shown in Equation 22.

Next, we deal with the :math:`h_t(\cdot)` in Equation 22.  Since we know
that :math:`\theta_t` did not vary much in our interval :math:`t \in (a_n, a_{n+1}]`
given our :math:`\epsilon_t << 1` assumption, we have :math:`h_t(\theta_t) = O(1)`
in our interval since our gradients vary smoothly (again due to 
`Lipschitz contiuity <https://en.wikipedia.org/wiki/Lipschitz_continuity>`__).
Additionally each :math:`h_t(\cdot)` will be a random variable which we can
assume to be independent, thus IID (doesn't change argument if they are
randomly partitioned which will only make the error smaller).  Plugging this
into :math:`\sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2} h_t(\theta_t)`, we
see the variance is :math:`O(\sum_{t=a_n+1}^{a_{n+1}} (\frac{\epsilon_t}{2})^2)`.
Putting this together in Equation 22, we get:

.. math::

   \sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2}\big(g(\theta_t) + h_t(\theta_t)\big)
   &= \frac{\epsilon_0}{2} g(\theta_{t=a_n}) + O(\epsilon_0) + O\Big(\sqrt{\sum_{t=a_n+1}^{a_{n+1}} (\frac{\epsilon_t}{2})^2}\Big) \\
   &= \frac{\epsilon_0}{2} g(\theta_{t=a_n}) + O(\epsilon_0) \\
   \tag{23}

From Equation 22, we can see the total stochastic gradient over our segment is
just the exact gradient starting from :math:`\theta_{t=a_n}` with step size
:math:`\epsilon_0` plus a :math:`O(\epsilon_0)` error term.  But recall our 
injected noise was of order :math:`O(\sqrt{\epsilon_0})`, which in turn dominates
:math:`O(\epsilon_0)` (for :math:`\epsilon_0 < 1`).  Thus for small
:math:`\epsilon_0`, our sequence :math:`\theta_{t=a_1}, \theta_{t=a_2}, \ldots`
will approximate LMC because each segment will essentially be an LMC update
with very decreasing small error.  As a result, this *subsequence* will
converge to the posterior as required.

--------------

Now the above argument showing that there exists a subsequence that samples
from the posterior isn't that useful because we don't know what that
subsequence is!  But [Teh2015]_ provides a much more rigorous treatment
of the subject showing a much more useful result in Theorem 7.  Without
going into all of the mathematical rigour, I'll present the basic idea 
(from what I can gather):

    **Theorem 1:** (Summary of Theorem 7 from [Teh2015]_)
    For a test function :math:`\varphi: \mathbb{R}^d \to \mathbb{R}`, the
    expectation of :math:`\varphi` with respect to the exact posterior
    distribution :math:`\pi` can be approximated by the weighted sum of
    :math:`m` SGLD samples :math:`\theta_0 \ldots \theta_{m-1}` that holds
    almost surely (given some assumptions):

    .. math::

        \lim_{m\to\infty} \frac{\epsilon_1 \varphi(\theta_0) + \ldots + \epsilon_m \varphi(\theta_{m-1})}{\sum_{t=1}^m \epsilon_t} = \int_{\mathbb{R}^d} \varphi(\theta)\pi(d\theta)
        \tag{24}

Theorem 1 gives us a more practical way to utilize the samples from SGLD.
We don't need to generate the exact samples that we would get from LMC,
instead we can just directly use the SGLD samples and their respective step sizes to
compute a weighted average for any actual quantity we would want (e.g.
expectation, variance, credible interval etc.).  According to Theorem 1,
this will converge to the exact quantity using the true posterior.
See [Teh2015]_ for more details (if you dare!).

Preconditioning
---------------

One problem both with SGD and SGLD is that the gradients updates might
be very slow due to the curvature of the loss surface.  This is known
to be a common phenomenon in large parameter models like neural networks
where there are many `saddle points <https://en.wikipedia.org/wiki/Saddle_point>`__.
These parts of a surface have very small gradients (in at least one dimension),
which will cause any SGD-based optimization procedure to be very slow.  On the
other end, if one of the dimensions in your loss has large curvature
(and thus gradient), it could cause unnecessary oscillations in one dimension
while the other one with low curvature crawls along.  The solution to this
problem is to use preconditioner.

.. figure:: /images/sgld-precondition.png
    :height: 250px
    :alt: Preconditioning
    :align: center

    **Figure 1: (Left) Original loss landscape, SGD converges slowly. 
    (Right) Transformed loss landscape with a preconditioner with reduced
    oscillations and faster progress.  Notice the counter lines are more evenly spaced
    out in each direction. (source:** [Dauphin2015]_ **)**

Preconditioning is a type of local transform that changes the optimization landscape
so the curvature is equal in all directions ([Dauphin2015]_).  As shown in Figure 1, preconditioning
can transform the curvature (shown by the contour lines) and as a result make SGD converge
more quickly.  Formally, for a loss function :math:`f` with parameters :math:`\theta \in \mathbb{R}^d`,
we introduce a non-singular matrix :math:`{\bf D}^{\frac{1}{2}}` such that :math:`\hat{\theta}={\bf D}^{\frac{1}{2}}\theta`.
Using the change of variables, we can define a new function :math:`\hat{f}(\hat{\theta})` that
is equivalent to our original function with its associated gradient (using the chain rule):

.. math::

    \hat{f}(\hat{\theta}) &= f({\bf D}^{-\frac{1}{2}}\hat{\theta})=f(\theta) \\
    \nabla\hat{f}(\hat{\theta}) &= {\bf D}^{-\frac{1}{2}}\nabla f(\theta)
    \tag{25}

Thus, regular SGD can be performed as such on the original :math:`\theta`, and for convenience,
we'll define :math:`{\bf G}={\bf D}^{-1}`:

.. math::

   \hat{\theta}_t &= \hat{\theta}_{t-1} - \epsilon \nabla \hat{f}(\hat{\theta}) \\
   \hat{\theta}_t &= \hat{\theta}_{t-1} - \epsilon {\bf D}^{-\frac{1}{2}}\nabla f(\theta) 
        && {Eq. } 25 \\
   \theta_t &= \theta_{t-1} - \epsilon {\bf D}^{-1}\nabla f(\theta) && \text{multiply through by } {\bf D}^{-\frac{1}{2}} \\
   \theta_t &= \theta_{t-1} - \epsilon {\bf G}(\theta_{t-1})\nabla f(\theta) && \text{rename } {\bf D}^{-1} \text{ to } {\bf G}\\
   \tag{26}

So the transformation turns out to be quite simple by multiplying our gradient
with a user chosen preconditioning matrix :math:`{\bf G}` that is usually a function of the
current parameters :math:`\theta_{t-1}`.  In the context of SGLD, we
have an equivalent result ([Li2016]_) where :math:`{\bf G}` defines a
Riemannian manifold:

.. math::

   \Delta \theta_t &= \frac{\epsilon_t}{2} \big[ {\bf G}(\theta_t) \big (\nabla \log[p(\theta_t)] + \frac{N}{n} \sum_{i=1}^n \nabla \log[p(x_{ti} | \theta_t)]\big) + \Gamma(\theta_t) \big] + {\bf G}^{\frac{1}{2}}(\theta_t)\varepsilon \\
        \varepsilon &\sim N(0, \epsilon_t)  \\
        \tag{27}

where :math:`\Gamma(\theta_t) = \sum_j \frac{\partial G_{i,j}}{\partial
\theta_j}` describe how the preconditioner changes with respect to
:math:`\theta_t`.  Notice the preconditioner is applied to the noise as well.

Previous approaches to use a preconditioner relied on the
expected 
`Fisher information <https://en.wikipedia.org/wiki/Fisher_information>`__
matrix, which is too costly for any modern deep learning model with many
parameters since it grows with the square of the parameters (similar to the
Hessian).  It turns out that we don't specifically need the Fisher information matrix,
we just need something that defines the Riemannian manifold metric, which only requires
a `positive definite matrix <https://en.wikipedia.org/wiki/Definite_matrix>`__.

The insight from [Li2016]_ was that we can use RMSprop as the preconditioning
matrix since it satisfies the positive definite criteria, and has shown
empirically to do well in SGD (being only a diagonal preconditioner matrix):

.. math::

   G(\theta_{t+1}) = diag\big(\frac{1}{\lambda + \sqrt{v(\theta_{t+1})}}\big) \tag{28}

where :math:`v(\theta_{t+1})=v(\theta, t)` is from Equation 14 and
:math:`\lambda` is a small constant to prevent numerical instability.

Additionally, [Li2016]_ has shown that there is no need to include the
:math:`\Gamma(\theta)` term in Equation 27 (even though it's not too hard to
compute for a diagonal matrix).  This is because it introduces an additional
bias term that scales with :math:`\frac{(1-\alpha)^2}{\alpha^3}` (from Equation 27), 
which is practically always set close to 1 (e.g. PyTorch's default for 
`RMSprop <https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html>`__ is :math:`0.99`).
As a result, we can simply use off-the-shelf RMSprop with only a slight
adjustment to the SGLD noise and gain the benefits of preconditioning.

Practical Considerations
------------------------

Besides preconditioning, SGLD has some other caveats inherited from MCMC.
First your initial condition matters, so you likely want to run it for a while
before you start sampling (i.e., "burn-in").  Similarly, adjacent samples
(particularly with a random walk method such as LMC/SGLD) will be highly
correlated so you will only want to take periodic samples to get (mostly)
independent samples (although with Theorem 1 this may not be necessary)
depending on your application.  Finally, for both deep learning and MCMC, your
hyperparameters matter a lot.  For example, initial conditions, learning rate
schedule, and priors all matter a lot.  So while a lot of the above techniques
help, there's no free lunch here.

Bayes by Backprop
=================

Bayes by Backprop ([Blundell2015]_) is a generalization of some previous work
to allow an approximation of Bayesian uncertainty, particularly for weights in
large scale neural network models where traditional MCMC methods do not scale.
Approximation is the key word here as it utilizes variational inference
(Equation 16).  That is, instead of directly estimating the posterior, it 
preselects the functional form of a distribution (:math:`q(\theta|\phi)`)
parameterized by :math:`\phi`, and optimizes :math:`\phi` using Equation 16.
The right hand side of Equation 16 is often called the *variational free
energy* (among other names), which we'll denote by :math:`\mathcal{F}(X, \phi)`:

.. math::

  \mathcal{F}(X, \phi) =  E_q\big[\log p(X|\theta) + \log p(\theta) - \log q(\theta|\phi)\big] 
  \tag{29}

Recall that instead of solving for point estimates of :math:`\theta`, we're
trying to solve for :math:`\phi`, which implicitly gives us (approximate)
distributions in the form of :math:`q(\theta|\phi)`.  To make this concrete,
for a neural network, :math:`\theta` would be the weights and instead of a
single number for each one, we would have a known distribution :math:`q(\theta|\phi)`
(that we select) parameterized by :math:`\phi`.

The main problem with Equation 29 is that we will need to sample from
:math:`q(\theta|\phi)` in order to approximate the expectation, but we will
also need to backprop through the "sample" in order to optimize :math:`\phi`.
If this sounds familiar, it is precisely the same issue we had with variation
autoencoders.  The solution there was to use the "reparameterization trick"
to rewrite the expectation in terms of a standard Gaussian distribution (and
some additional transformations) to yield an equivalent loss function that we
can backprop through.  

As you may expect, [Blundell2015]_ generalizes this concept beyond Gaussians
to any distribution with the following proposition:

    **Proposition 1:** (Proposition 1 from [Blundell2015]_)
    Let :math:`\varepsilon` be a random variable with probability density
    given by :math:`q(\varepsilon)` and let :math:`\theta = t(\phi, \varepsilon)`
    where :math:`t(\phi, \varepsilon)` is a deterministic function.
    Suppose further that the marginal probability density of :math:`\theta`,
    :math:`q(\theta|\phi)`, is such that 
    :math:`q(\varepsilon)d\varepsilon = q(\theta|\phi)d\theta`.  Then for a function
    :math:`f(\cdot)` with derivatives in :math:`\theta`:

    .. math::
    
       \frac{\partial}{\partial\phi}E_{q(\theta|\phi)}[f(\theta,\phi)] =
       E_{q(\varepsilon)}\big[
        \frac{\partial f(\theta,\phi)}{\partial\theta}\frac{\partial\theta}{\partial\phi}
            + \frac{\partial f(\theta, \phi)}{\partial \phi}
       \big]
       \tag{30}

    **Proof:**

    .. math::

       \frac{\partial}{\partial\phi}E_{q(\theta|phi)}[f(\theta,\phi)]
           &= \frac{\partial}{\partial\phi}\int f(\theta,\phi)q(\theta|\phi)d\theta \\
           &= \frac{\partial}{\partial\phi}\int f(\theta,\phi)q(\varepsilon)d\varepsilon && \text{Given in proposition}\\
           &= \int \frac{\partial}{\partial\phi}[f(\theta,\phi)]q(\varepsilon)d\varepsilon \\
           &= E_{q(\varepsilon)}\big[
           \frac{\partial f(\theta,\phi)}{\partial\theta}\frac{\partial\theta}{\partial\phi}
               + \frac{\partial f(\theta, \phi)}{\partial \phi}\big]  && \text{chain rule}
          \\
       \tag{31}

So Proposition 1 tells us that the "reparameterization trick" is valid in the context of 
gradient based optimization (i.e., SGD) if we can show 
:math:`q(\varepsilon)d\varepsilon = q(\theta|\phi)d\theta`.
Equation 30 may be a bit cryptic because of all the partial derivatives but notice two things. 
First, the expectation is no with respect to a standard distribution :math:`q(\varepsilon)`,
and, second, the inner part of the expectation is done automatically through backprop when
you implement :math:`t(\phi, \varepsilon)` so you don't have to explicitly calculate it
(it's just the chain rule though).  Let's take a look at a couple of examples.

First, let's take a look at the good old Gaussian distribution with parameters
:math:`\phi = \{\mu, \sigma\}` and :math:`\varepsilon` being a standard Gaussian.
We let :math:`t(\mu, \sigma, \varepsilon) = \sigma \cdot \varepsilon + \mu`.
Thus, we have:

.. math::

   q(\theta | \mu, \sigma)d\theta 
       &= \frac{1}{\sqrt{2\pi\sigma^2}}\exp\{-\frac{(\theta - \mu)^2}{2\sigma^2}\}d\theta && \text{Gaussian pdf} \\
       &= \frac{1}{\sqrt{2\pi\sigma^2}}\exp\{-\frac{((\sigma \cdot \varepsilon + \mu)- \mu)^2}{2\sigma^2}\}\sigma d\varepsilon && \theta = \sigma \cdot \varepsilon + \mu \\
       &= \frac{1}{\sqrt{2\pi}}\exp\{-\frac{\varepsilon^2}{2}\} d\varepsilon \\
       &= q(\varepsilon)d\epsilon
       \tag{32}
            
We can easily see that the two expressions are the same.  To drive the point home,
we can show the same relationship with the exponential distribution parameterized by :math:`\lambda`
using :math:`t(\lambda, \varepsilon) = \frac{\varepsilon}{\lambda}` for standard exponential
distribution :math:`\varepsilon`:

.. math::

   q(\theta | \lambda)d\theta 
       &= \lambda \exp\{-\lambda \theta\}d\theta && \text{Exponential pdf} \\
       &=\lambda \exp\{-\lambda \frac{\varepsilon}{\lambda}\}\frac{d\varepsilon}{\lambda} && \theta = \frac{\varepsilon}{\lambda} \\
       &= \exp\{-\varepsilon\}d\varepsilon \\
       &= q(\varepsilon)d\epsilon
       \tag{33}

The nice thing about this trick is that it's widely implemented in modern tooling.
For example PyTorch has an implementation on distributions where this condition is true
using the `rsample()` method.  You can look into each of the respective implementations to
see how the :math:`t(\cdot)` function is defined.  See 
`Pathwise derivative <https://pytorch.org/docs/stable/distributions.html#pathwise-derivative>`__
section of the PyTorch docs for details.

With these this reparameterization trick (and picking appropriate distributions), one
can easily implement variational inference by substituting the exact posterior for
a fixed parameterized distribution (e.g., Gaussian, exponential etc.).  This allows
you to easily train the network using standard SGD methods
that sample from this approximate posterior distribution but *importantly* can
backprop through them to update the parameters of these approximate posteriors
to hopefully achieve a good estimate of uncertainty.  Note however that variational
inference will often `underestimate variance <https://www.quora.com/Why-and-when-does-mean-field-variational-Bayes-underestimate-variance>`__.
So there's also no free lunch here either.

Experiments
===========

Simple Gaussian Mixture Model
-----------------------------

The first experiment I did was try to reproduce the simple mixture model with
tied means from [Welling2011]_.  The model from the paper is specified as:

.. math::

    \pi &\sim Bernoulli(p) \\
    \theta_1 &\sim \mathcal{N}(0, \sigma_1^2) \\
    \theta_2 &\sim \mathcal{N}(0, \sigma_2^2) \\
    x_i &\sim \pi * \mathcal{N}(\theta_1, \sigma_x^2) + (1-\pi) * \mathcal{N}(\theta_1 + \theta_2, \sigma_x^2) \\
    \tag{34}

with :math:`p=0.5, \sigma_1^2=10, \sigma_2^2=1, \sigma_x^2=2`.  They generate
100 :math:`x_i` data points using a fixed :math:`\theta_1=0, \theta_2=1`.
In the paper, they say that this generates a bimodal distribution but I wasn't
able to reproduce it.  I had to change :math:`\sigma_x^2=2.56` to get a slightly wider distribution to get something
similarly bimodal.  I did this *only* for the data generation, all the other 
estimation below uses :math:`\sigma_x^2=2`.  Theoretically, if they got a weird
random seed they might be able to get something bimodal, but I wasn't able to.
Figure 2 shows a histogram of the data I generated with the modified
:math:`\sigma_x^2=2.56`.


.. figure:: /images/sgld-mixture_hist.png
    :height: 350px
    :alt: mixture hist
    :align: center

    **Figure 2: Histogram of** :math:`x_i` **datapoints**

From Equation 34, you can that the only parameters we need to estimate are
:math:`\theta_1` :math:`\theta_2`.  If our procedure is correct, we would
our posterior distribution to have a lot of density around 
:math:`(\theta_1, \theta_2) = (0, 1)`.  

.. figure:: /images/sgld-mixture-exact.png
    :height: 450px
    :alt: mixture exact
    :align: center

    **Figure 3: True posterior**

Since this is just a relatively simple two dimensional problem, you can
estimate the posterior by discretizing the space and calculating the
unnormalized posterior (likelihood x prior) for each cell.  As long as you
don't overflow your floating point variables, you should be able to get a
contour plot as shown in Figure 3.  As you can see, the distribution is bimodal
with a peak at :math:`(-0.25, 1.5)` and :math:`(1.25, -1.5)`.  It's not exactly
the :math:`(0, 1)` peak we were expecting, but considering that we only sampled
100 points, this is the "best guess" based on the data we've seen.

Results
_______

The first obvious thing to do is estimate the posterior using MCMC.  I used
`PyMC <https://www.pymc.io/welcome.html>`__ for this because I think it has the
most intuitive interface.  The code is only a handful of lines and is made easy 
with the builtin `NormalMixture` distribution.  I used the default NUTS sampler
(extension of HMC) to generate 5000 samples with a 2000 sample burnin.
Figure 4 shows the resulting contour plot, which line up very closely with the
exact results in Figure 3.

.. figure:: /images/sgld-mixture_mcmc.png
    :height: 450px
    :alt: mixture mcmc
    :align: center

    **Figure 4: MCMC estimate of posterior**

Lastly, I implemented both SGD and SGLD in PyTorch (using the same PyTorch
Module).  This was pretty simple by leveraging the builtin `distributions
<https://pytorch.org/docs/stable/distributions.html>`__ package, particularly
the `MixtureSameFamily <https://pytorch.org/docs/stable/distributions.html>`__
one.  

For SGD with batch size of :math:`100` and learning rate (:math:`\epsilon`)
0.01 and 300 epochs with initial values as :math:`(\theta_1, \theta_2) = (1,
1)`, I was able to iterate towards a solution of :math:`(-0.2327, 1.5129)`,
which is pretty much bang on the first mode.  This gave me confidence that
my model was correct.  

Next, moving onto SGLD, I used the same effective learning rate schedule as the
paper with :math:`a=0.01, b=0.0001, \gamma=0.55` that results in 10000 sweeps
through the entire dataset with batch size of 1.  I also did different
experiments with batch size of 10 and 100, adjusting the same decaying
polynomial schedule so that the total number of gradient updates are the same
(see the `notebook <https://github.com/bjlkeng/sandbox/blob/master/stochastic_langevin/normal_mixture.ipynb>`__).
I didn't do any burnin or thinning (although I probably should have?).
The results are shown in Figure 5.

.. figure:: /images/sgld-mixture_sgld.png
    :height: 650px
    :alt: mixture slgd
    :align: center

    **Figure 5: HMC and SGLD estimates of posterior for various batch sizes**

We can see that SGLD is no panacea for posterior estimation.  With batch size of 100,
it only ever explores one mode.  Likely, I would have to play with the learning
rate/schedule to ensure that it starts high enough that the Langevin dynamics
will let it wander to the other mode.  Considering I started at :math:`(1,1)`,
it's no surprise that it drifted towards the top left first.  The upside is that
it seemed to be squarely centred on one of the true modes that SGD found at
:math:`(-0.25, 1.5)`.

Batch size of 10 shows quite a different story.  It seemed to properly explore
the first mode but then wanders to the second mode and get stuck there.  Again,
we're seeing the sensitivity of SGLD to the learning rate/schedule.  The peak
on the second mode seems a bit off as well.  I should note that as mentioned in
the SGLD section, the samples from it are not guaranteed to match the true
posterior (theoretically only a subsequence is guaranteed).  So this comparison
of contour plots isn't exactly fair but we're looking at macro characteristics of
finding all the modes, which we would expect to see.

Lastly using a batch size of 1 (same as [Welling2011]_), we see something
closer to the true posterior with a clearly defined mode in the top left
corner, and a visible but less clearly defined mode in the bottom right.
Again, the story is likely that it wandered into the bottom right at some
point, but got stuck in the top left corner after a while.  This is kind of
expected as you shrink :math:`\epsilon`, it's just very unlikely to jump too
far away from the first mode it found.  The peaks of the samples are also off
from the exact posterior for the same reason as discussed.

My conclusion from this experiment is that vanilla SGLD is not a very robust
algorithm.  It's so sensitive to the learning rate, which can cause it to have
issues finding modes as seen above.  There are numerous extensions to SGLD that
I haven't really looked at (including ones that are inspired by HMC) so those
may provide more robust algorithms to do at scale posterior sampling.  Having
said that, perhaps you aren't too interested in trying to generate the exact
posterior.  In those cases, SGLD seems to do a *good enough* job at estimating
the uncertainty around one of the modes (at least in this simple case).

Stochastic Volatility Model
---------------------------

The next experiment I did was with a stochastic volatility model from the 
`example <https://www.pymc.io/projects/examples/en/latest/case_studies/stochastic_volatility.html>`__
in the PyMC docs.  This is actually kind of the opposite of what you would
want to use SGLD and Bayes by Backprop for because it is a complex model for
stock prices with a *single* time series, which is the observed price of the
S&P 500.  I mostly picked this model because I was curious how we could apply
these methods to more complex hierarchical Bayesian models.  Being one of the
prime examples of where Bayesian methods can be used to analyze a problem,
I naively thought that this would be an easy thing to model.  It turned out to
be much more complex than I expected as we shall see.

First, let's take a look at the definition of the model:

.. math::
   
   \sigma &\sim Exponential(10), & \nu &\sim Exponential(.1) \\
   s_0 &\sim Normal(0, 100), & s_i &\sim Normal(s_{i-1}, \sigma^2) \\
   \log(r_i) &\sim t(\nu, 0, \exp(-2 s_i)) \\
   \tag{35}

Equation 35 models the logarithm of the daily returns, :math:`r_i` with a 
`student-t distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`__,
parameterized by the degrees of freedom :math:`\nu` following an 
`exponential distribution <https://en.wikipedia.org/wiki/Exponential_distribution>`__,
and volatility :math:`s_i` where :math:`i` is the time index.  The volatility
follows a 
`Gaussian random walk <https://en.wikipedia.org/wiki/Random_walk#Gaussian_random_walk>`__ 
across all 2905 time steps, which is parameterized by a common variance given by an 
`exponential distribution <https://en.wikipedia.org/wiki/Exponential_distribution>`__.
To be clear, we are modeling the entire time series at once with a different
log-return and volatility random variable for each time step.
Figure 6 shows the model using `plate notation <https://en.wikipedia.org/wiki/Plate_notation>`__:

.. figure:: /images/sgld-vol_model.png
    :height: 400px
    :alt: vol model
    :align: center

    **Figure 6: Stochastic volatility model described using plate notation (** `source <https://www.pymc.io/projects/examples/en/latest/case_studies/stochastic_volatility.html>`__ **)**

This is a relatively simple model for explaining asset prices.  It is obviously
too simple to actually model stock prices.  One thing to point out is that we
have a single variance (:math:`\sigma`) of the volatility process across all
time.  This seems kind of unlikely given that we know different market regimes
will behave quite differently.  Further, I'm always pretty suspicious of 
Gaussian random walks.  This implies some sort of 
`stationarity <https://en.wikipedia.org/wiki/Stationary_distribution>`__, which 
obviously is not true over long periods of time (this may be an acceptable
assumption at very short time periods though).  In any case, it's a toy
hierarchical model that we can use to test our two Bayesian learning methods.

Modelling the Hierarchy
_______________________

The first thing to figure out is how to model Figure 6 using some combination
of our two methods.  Initially I naively tried applying SGLD directly but came
across a major issue: how do I deal with the volatility term :math:`s_i`?
Naively applying SGLD means instantiating a parameter for each random variable
you want to estimate uncertainty for, then applying SGLD using a standard
gradient optimizer.  Superficially, it looks very similar to using gradient
descent to find a point estimate.  The big problem with this approach is that 
the volatility :math:`s_i` is conditional on the step size :math:`\sigma`.
If we naively model :math:`s_i` as a parameter, it loses its dependence on
:math:`\sigma` and are unable to represent the model in Figure 6.  

It's not clear to me that there is a simple way around it using vanilla SGLD.
The examples in [Welling2011]_ were non-hierarchical models such as Bayesian
logistic regression that just needed to model uncertainty of the model
coefficients.  After racking my brain for a while on how to model it, I 
remembered that there was another example that I knew of for getting gradients
to flow through a latent variable -- variational autoencoders!  Yes, the good
old reparameterization trick comes to save the day.  This led me to the work on
this generalization in [Blundell2015]_ and one of the ways you estimate
uncertainty in Bayesian neural networks.

Let's write out some equations to make things more concrete. First the
probability model defining the notation :math:`x_i = \log(r_i)` for clarity:

.. math::

    p({\bf s}, \nu, \sigma | {\bf x}) &= [\Pi_1^N p(x_i | s_i, \nu, \sigma) p(s_i | s_{i-1}, \sigma)]p(s_0)p(\nu) p(\sigma) \\
    \\
    p(x_i | s_i, \nu, \sigma) &\sim t(\nu, 0, exp(s_i)) \\
    p(s_i|s_{i-1}, \sigma) &\sim N(s_{i-1}, \sigma^2) = N(0, \sigma^2) + s_{i-1} \\
    p(\sigma) &\sim Exp(10) \\
    p(\nu) &\sim Exp(0.1) \\
    \tag{36}

Notice the random walk of the stochastic volatility :math:`s_i` can be
simplified by pulling out the mean, so we only have to worry about the
additional zero-mean noise added at each step.  

.. admonition:: Why explicitly model :math:`s_i` uncertainty at all?

    One question you might ask is why do we need to explicitly model the
    uncertainty of :math:`s_i` at all?  Can't we just model :math:`\sigma` 
    (and :math:`\nu`) and then apply SGLD, sampling the implied value of
    :math:`s_i` along the way?  Well it turns out that this doesn't quite work.

    Naively for SGLD on the forward pass, you have a value for :math:`\sigma`,
    you can sample :math:`s_i = s_{i-1} + \sigma \cdot \varepsilon` where
    :math:`\varepsilon \sim N(0, 1)`, then propagate and compute the associated
    t-distributed loss for :math:`x_i`.  Similarly, you can easily backprop
    through this network since each computation is differentiable.

    Unfortunately, this does not correctly capture the uncertainty specified in
    :math:`s_i`.  One way to see this is that the sample we get using this
    method is :math:`s_i = s_0 + \sum_{i=1}^{i} \sigma \varepsilon`.  This is
    just a random walk with standard deviation :math:`\sigma` and starting
    point :math:`s_0`.  Surely, the posterior of :math:`s_i` is not just a
    scaled random walk.  This would completely ignore the observed values of
    :math:`x_i`, which would only affect the value of :math:`\sigma` (and
    :math:`\nu`).

    Another intuitive argument is that SGLD explores the uncertainty by
    "traversing" through the parameter space.  Similar to more vanilla MCMC
    methods, it should spend more time in high density areas and less time in
    low density ones.  If we are not "remembering" the values of :math:`s_i`
    via parameters, then SGLD cannot correctly sample from the posterior
    distribution since it cannot "hang out" in high density regions of
    :math:`s_i`.  That is why we need to both be able to properly model the 
    uncertainty of :math:`s_i` while still being able to backprop through it.

To deal with the hierarchical dependence of :math:`s_i` on :math:`\sigma`, we
approximate the posterior of :math:`s_i` using a Gaussian with learnable mean
:math:`\mu_i` and :math:`\sigma` as defined above:

.. math::

    p(s_i|s_{i-1},\sigma, {\bf x}) \approx q(s_i|s_{i-1}, \sigma; \mu_i) &= s_{i-1} + N(\mu_i, \sigma)  \\
    &= s_{i-1} + \sigma \varepsilon + \mu_i, &\varepsilon &\sim N(0, 1)\\
    \tag{37}

Notice that :math:`q` is not conditioned on :math:`\bf x`.  In other words, we are
going to use :math:`\bf x` (via SGLD) to estimate the parameter :math:`\mu_i`,
but there is no probabilistic dependency on :math:`\bf x`.  Next using the ELBO
from Equation 16, we want to be able to derive a loss to optimize our
approximate posterior :math:`q(s_i|s_{i-1}, \sigma; \mu_i)`:

.. math::

    \log p({\bf x}| s_0, \sigma, \nu) 
    &\geq -E_q[\log\frac{q(s_{1\ldots n}|s_0, \sigma, \mu_i)}{p({\bf s_{1\ldots n}, x}| s_0, \sigma, \nu)}] \\
    &= E_q[\sum_{i=1}^n \log p(s_i, x_i|s_{i-1}, \sigma, \nu) - \log q(s_i|s_{i-1}, \sigma, \mu_i)] \\
    &= E_q[\sum_{i=1}^n \log p(x_i|s_i, \nu) + \log p(s_i | s_{i-1}, \sigma) - \log q(s_i|s_{i-1}, \sigma, \mu_i)]
    \tag{38}

Finally, putting together our final loss based on the posterior we have:

.. math::

   \log p(s_0, \sigma, \nu| {\bf x}; {\bf \mu}) &\propto \log p(s_0, \sigma, \nu, {\bf x}; {\bf \mu}) \\
   &= \log p({\bf x} | s_0, \sigma, \nu; {\bf \mu}) + \log p(s_0) + \log p(\sigma) + \log p(\nu)  \\
   &\approx E_q[\sum_{i=1}^n \log p(x_i|s_i, \nu) + \log p(s_i | s_{i-1}, \sigma) - \log q(s_i|s_{i-1}, \sigma, \mu_i)] \\
   &\hspace{10pt} + \log p(s_0) + \log p(\sigma) + \log p(\nu)  \\
   \tag{39}

We can see from Equation 39, that we have likelihood terms (:math:`\log p(x_i|s_i, \nu)`, 
:math:`\log p(s_i | s_{i-1}, \sigma)`), prior terms (:math:`\log p(s_0)`,
:math:`\log p(\sigma)`, :math:`\log p(\nu)`), and a regularizer from our variational
approximation (:math:`\log q(s_i|s_{i-1}, \sigma, \mu_i)`).  This is a common
pattern in variational approximations with an ELBO loss.

With the loss we have enough to (approximately) model our stochastic volatility problem.
First, start by defining a learnable parameter for each of :math:`\sigma, \nu, s_0, \mu_i`.
Next, the forward pass is simply computing the :math:`s_i` values using the
reparameterization trick in Equation 37 using the loss from Equation 39.  Only
a minor adjustment to SGD to change it in the SGLD and you are off to the races!

An important point to make this practically train was to implement the RMSprop
preconditioner from Equation 28.  Without it I was unable to get a reasonable fit.
This is probably analogous to most deep networks: if you don't use a modern
optimizer, it's really difficult to fit a deep network.  In this case we're
modeling more than 2900 time steps, which can cause lots of issues when
backpropagating.

Results
_______

The first thing to look at are the results generated using HMC via PyMC, whose code
was taken directly from the 
`example <https://www.pymc.io/projects/examples/en/latest/case_studies/stochastic_volatility.html>`__.
Figure 7 shows the posterior :math:`\sigma` and :math:`\nu` for two chains (two
parallel runs of HMC).  :math:`\sigma` (step size) has a mode around 0.09 -
0.10 while :math:`\nu` has a mode between 9 and 10.  Recall that these variables 
parameterize and an `exponential distribution <https://en.wikipedia.org/wiki/Exponential_distribution>`__, 
so the expected value of the corresponding random  variables are :math:`\sigma
\approx 10` and :math:`\nu \approx 0.1` (the inverse of the posterior).

.. figure:: /images/sgld_mcmc_stepsize.png
   :height: 350px
   :align: center
   
   **Figure 7: HMC posterior estimate of** :math:`\sigma, \nu` **using PyMC**

The more interesting distribution is the volatility shown in Figure 8.  Here we see that there
are certain times with high volatility such as 2008 (the financial crisis).
These peaks in volatility also have higher uncertainty around them (measured by
the vertical width of the graph), which matches our intuition that higher
volatility usually means unpredictable markets making the volatility itself
hard to estimate.

.. figure:: /images/sgld_mcmc_vol.png
    :height: 350px
    :align: center

    **Figure 8: HMC posterior estimate of the volatility**

The above stochastic volatility model was implemented using a simple PyTorch model
Module and builtin the `distributions <https://pytorch.org/docs/stable/distributions.html>`__
package doing a lot of the heavy work.  I used a mini-batch size of 100 even
though I only had once trace by repeating it 100 times.  I found that this
stabilized the gradient estimates from the Gaussian sampled :math:`\b s`
values.  The RMSprop preconditioner was quite easy to implement by inheriting
from the existing PyTorch class and overriding the :math:`step()` function (see
the notebook).  I used a burnin of 500 samples with a fixed starting learning rate of
0.001 throughout the burnin after which the decayed polynomial learning rate
schedule kicks in.  I didn't use any thinning.  Figure 9 shows the estimate for
:math:`\sigma` and :math:`\nu` using SGLD.  

.. figure:: /images/sgld_sgld_sigma_nu.png
    :height: 300px
    :align: center

    **Figure 9: Posterior estimate of ** :math:`\sigma, \nu` **using SGLD**

Starting with :math:`\nu`, its mode is not too far off with a value around
:math:`9.75`, however the width of the distribution is much tighter with most
of the density in between 9.7 and 9.8.  Clearly either SGLD and/or our
variational approximation has changed the estimate of the degrees of freedom.  

This is even more pronounced with :math:`\sigma`.
Here we get a mode around 0.025, which is quite different than the 0.09 - 0.10
we saw above with HMC.  However, recall we are estimating parameters of a
different model with :math:`\sigma` is parameterizing the variance our
approximate posterior, so we would expect that it wouldn't necessarily capture
the same value.  This points out a limitation of our approach: our parameter
estimates in the approximate hierarchical model will not necessarily be
comparable to the exact one.  Thus, we don't necessarily get the
interpretability of the model that we would expect in a regular Bayesian
statistics flow.

.. figure:: /images/sgld_sgld_vol.png
    :height: 350px
    :align: center

    **Figure 10: Posterior estimate of the stochastic volatility via SGLD of the approximate posterior mean**

Finally, Figure 10 shows the posterior estimate of the stochastic volatility :math:`\bf s`.
Recall, that we approximated :math:`s_i \approx q(\mu_i, \sigma) \sim N(\mu_i, \sigma)`.
However, we cannot use :math:`q(\mu_i, \sigma)` directly to estimate the
volatility because that would mean the variance of the volatility at each
timestep :math:`s_i` would be equal, which clearly it is not.  Instead, I used
SGLD to estimate the distribution of each :math:`\mu_i` and plotted that
instead.  Interestingly, we get a very similar shaped time series but with
significantly less variance at each time step.  For example, during 2008
the variance of the volatility hardly changes staying close to 0.04, whereas in
the HMC estimate it's much bigger swinging from almost 0.035 to 0.08.

One reason that we see lower variance that is often cited is that variational
inference often underestimates the 
`variance <https://www.quora.com/Why-and-when-does-mean-field-variational-Bayes-underestimate-variance>`__.
This is because it is optimizing the KL divergence between the approximate
posterior :math:`q` and the exact one :math:`p`.  This means that this is
more likely to favour low variance estimates, see my `previous post <link://slug/semi-supervised-learning-with-variational-autoencoders>`__ for more details.
Another (perhaps more likely?) reason is that the approximation is just not a
good one.  Perhaps a more complex joint distribution across all :math:`s_i` is
what is really needed given the dependency between them.  In any case, it points
to the difficulty plugging these tools into a more typical Bayesian statistics
workflow (which they were not at all intended to be used for by the way!).

Implementation Notes
--------------------

Here are some unorganized notes about implementing the above two toy experiments.
As usual, you can find the corresponding 
`notebooks on Github <https://github.com/bjlkeng/sandbox/blob/master/stochastic_langevin/>`__.

* In general, implementing SGLD is quite simple.  Literally you just need to
  add a noise term to the gradient and update as usual in SGD.  Just be careful
  that the *variance* of the Gaussian noise is equal to the learning rate
  (thus standard deviation is the square root of that).
* The builtin `distributions <https://pytorch.org/docs/stable/distributions.html>`__ package 
  in PyTorch is great.  It's so much less error prone than writing out the log density yourself
  and it has so many nice helper functions like `rsample()` to do reparameterized sampling
  and `log_prob()` to compute the log probability.
* The one thing that required some careful coding was adding mini-batches to
  the stochastic volatility model.  It's nothing that complicated but you have to ensure
  all the dimensions add up and you setting up your PyTorch distributions to
  have the correct dimension.  Generally, you'll want one copy of the parameters but
  replicate them when you are computing forward/backward and then average over
  your batch size in your loss.
* For computing the mixture distributions in the first experiment, I carelessly just
  took the weighted average of two Gaussians log densities -- this is not correct!
  The weighted average needs to be done in non-log space and then logged.  Alternatively,
  it's just much easier to use the builtin PyTorch function of `MixtureSameFamily()`
  to do what you need.
* One silly (but conceptually important) mistake was getting PyTorch scalars
  (i.e., zero dimensional tensors) and one dimensional tensors (i.e., vectors)
  with one element confused.  Depending on the API, you're going to want one or the other
  and need to use `squeeze()` or `unsqueeze()` as appropriate.
* Don't forget to use `torch.no_grad()` in your optimizer or else PyTorch will try
  to compute the computational graph of your gradient updates and cause an error.
* For the brute force computation to estimate the exact posterior for the Gaussian mixture,
  you need to compute the unnormalized log density for a grid and the
  exponentiate it to get the probability.  Obviously exponentiating it can
  cause overflow, so I scaled the unnormalized log density by subtracting the
  max value and then exponentiate.  Got to pay attention to numerical stability sometimes!
* For the stochastic volatility model for time step :math:`s_i`, the naive random walk
  posterior that we considered (by not modelling it at all) would cause the variance
  at :math:`Var(s_i) = \sum_{j=1}^i Var(s_j)`.  This is because a random walk is a sum
  of independent random variables, meaning the sum at the :math:`i^{th}` step
  is the sum of the variances.  This is obviously not what we want.
* I had to set the initial value of :math:`s_0` close to the value of the
  posterior mean of :math:`s_1` or else I didn't get something to fit well.  I
  suspect that it's just really hard to backprop so far back and move the value
  of :math:`s_0` significantly.
* On that topic, I initialized :math:`\sigma, \nu` to the means of the
  respective priors and :math:`\bf s` to a small number near 0.  Both of these
  seemed like reasonable choices.
* I had to tune the stochastic volatility model to get a fit like you see above.
  Too little and it wouldn't get the right shape.  Too much and it would get a
  strange shape as well with :math:`\sigma` continually shrinking.  I suspect
  the approximate Gaussian posterior is not really a good fit for this model.
* While implementing the RMSprop preconditioner, I used inherited from the
  PyTorch implementation and overrode `step()` function.  Using that function
  as a base, it's interesting to see all the various branches and special cases
  it handles beyond the vanilla one (e.g. momentum, centered, weighted_decay).
  Of course in my implementation I just ignored all of them and only
  implemented the simplest case but makes you appreciate the extra work that needs
  to be done to write a good library.
* I added a random seed at some point in the middle just so I could reproduce
  my results.  This was important because of the high randomness from the Bayes
  by Backprop sampling in the training.  Obviously it's good practice but when
  you're just playing around it's easy to ignore.

Conclusion
==========

Another post on an incredibly interesting topic.  To be honest, I'm a bit
disappointed that it was some magical solution to doing Bayesian learning but
it makes sense that it is not because otherwise all the popular libraries would
have already implemented it.  The real reason I got onto this topic is because
it is important conceptually to a stream of research that I've been trying to
build up to.  I find it incredibly satisfying to learn things "from the ground
up", going back to the fundamentals.  I feel that this is the best way to get a
strong intuition for the techniques.  The downside is that you go down so many
rabbit holes and don't make too much direct progress towards a target.
Fortunately, I'm not beholden to any sort of pressures like academics so I can
wander around to my hearts content.  As they say, it's about the journey not
the destination.  See you next time!

References
==========
* Previous posts: `Markov Chain Monte Carlo and the Metropolis Hastings Algorithm  <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__, `Hamiltonian Monte Carlo <link://slug/hamiltonian-monte-carlo>`__, `The Expectation Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__, `Variational Autoencoders <link://slug/variational-autoencoders>`__, `An Introduction to Stochastic Calculus <link://slug/an-introduction-to-stochastic-calculus>`__

.. [Welling2011] Max Welling and Yee Whye Teh, "`Bayesian Learning via Stochastic Gradient Langevin Dynamics <https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf>`__", ICML 2011.
.. [Blundell2015] Blundell et. al, "`Weight Uncertainty in Neural Networks <https://arxiv.org/abs/1505.05424>`__", ICML 2015.
.. [Li2016] Li et. al, "`Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks <https://arxiv.org/abs/1512.07666>`__", AAAI 2016.
.. [Radford2012] Radford M. Neal, "MCMC Using Hamiltonian dynamics", `arXiv:1206.1901 <https://arxiv.org/abs/1206.1901>`__, 2012.
.. [Teh2015] Teh et. al, "Consistency and fluctations for stochastic gradient Langevin dynamics", `arXiv:1409.0578 <https://arxiv.org/abs/1409.0578>`__, 2015.
.. [Dauphin2015] Dauphin et. al, "Equilibrated adaptive learning rates for non-convex optimization", `arXiv:1502.04390 <https://arxiv.org/abs/1502.04390>`__, 2015.
