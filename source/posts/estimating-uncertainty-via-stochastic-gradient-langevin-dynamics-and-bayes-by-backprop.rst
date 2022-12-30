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
my heart: Bayesian learning and scalability.  I dare ask wouldn't be interested
in the intersection of such topics?  In any case, I hope you enjoy my
explanation of it.

This post is about two techniques to perform scalable Bayesian inference.  They
both address the problem using stochastic gradient descent (SGD) but in very
different ways.  One leverages the observation that SGD plus some noise will
converge to Bayesian posterior sampling [Welling2011]_, while the other generalizes the
"reparameterization trick" from variational autoencoders to enable non-Gaussian
posterior approximations [Blundell2015]_.  Both are easily implemented in the modern deep
learning toolkit thus benefit from the massive scalability of that toolchain.
As usual, I go over the necessary background (or refer you to my previous
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
distribution of the statistical parameters of your model, which in turns allows
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

Generally, there is no closed form and you have to resort to heavy methods such
as Markov Chain Monte Carlo (MCMC) methods or some form of approximation (which
we'll get to later).  MCMC never give you an exact closed form but instead give
you either samples from the posterior distribution, which you can use then use
to compute any statistics you like.  These methods are quite slow because they
rely on `Monte Carlo <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__
methods, which require repeated random sampling. 

This brings us to our first scalability problem Bayesian learning: it does not
scale well with the number of parameters.  Randomly sampling with MCMC implies
that you have to "walk" the parameter space, which potentially grows
exponentially with the number of parameters.  There are many techniques to make
this more efficient but ultimately it's hard to compensate for an exponential.
The natural model for this situation is neural networks which can have orders
of magnitude more parameters compared to classic Bayesian learning problems
(I'll also add that the use-case of the posterior is usually different too).

The other non-obvious scalability issue with MCMC from Equation 1 is the data.
Each evaluation of MCMC requires an evaluation of the likelihood and prior from
Equation 1.  For large data (e.g. modern deep learning datasets), you quickly
hit issues either with memory and/or computation speed.

Modern deep learning has really solved both of these problems by leveraging the
one of the simplest optimization method out there (stochastic gradient descent)
along with the massive compute power of modern hardware (and its associated
toolchain).  How can we leverage these developments to scale Bayesian learning?
Keep reading to found out!

Background
==========

Bayesian Networks and Bayesian Hierarchical Models
--------------------------------------------------

We can take the idea of parameters and prior from Equation 1 to multiple
levels.  Equation 1 implicitly assumes that there is one "level" of parameters
(:math:`\theta`) that we're trying to estimate with prior distributions
(:math:`p({\bf \theta})`) attached to them, but there's no reason why you only
need a single level.  In fact, our parameters can be conditioned on parameters,
which can be conditioned on parameters, and so on.  
This is called `Bayesian hierarchical modeling <https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling>`__.
If this sounds oddly familiar, it's the same thing as `Bayesian networks
<https://en.wikipedia.org/wiki/Bayesian_network#Graphical_model>`__ in a different context (if you're
familiar with that).  My `previous post <link://slug/the-expectation-maximization-algorithm>`__ that gives a nice high
level summary on the intuition with latent variables.

To quickly summarize, in a parameterized statistical model there are broadly
two types of variables: observed and unobserved.  Observed are the ones
where we have values for often with multiple observations where we assume
they are `IID <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`__.

Unobserved variables can have different names. In Bayesian networks they
are usually called latent or hidden (random) variables, which can have 
complex conditional dependencies specified as a DAG.  In hierarchical models
they are called **hyperparameters**, which are the parameters of the 
observed models, the parameters of parameters, parameters of parameters of
parameters and so on.  Similarly, each of these hyperparameters has a 
distribution which we call a **hyperprior**.  

These two concepts are mathematically the same and from what I gather really
on vary based on the context.  In the context of hierarchical models,
the hyperparameters and hyperpriors represent some structural knowledge
about the problem, hence of the use of term "priors".  The data is typically
believed to appear in hierarchical "clusters" that share similar attributes
(i.e., drawn from the same distribution).  This view is more typical in
Bayesian statistics applications where the number of stages (and thus
variables) is usually small (two or three).  If terms such as 
`fixed or random effects models <https://en.wikipedia.org/wiki/Multilevel_model>`__, 
ring a bell, then this framing will make much more sense.

In Bayesian networks, the latent variables can represent the underlying
phenomenon but also can be artificially introduced to make the problem more
tractable.  This happens more often in machine learning e.g. `variational
autoencoders <link://slug/variational-autoencoders>`__.  In these contexts,
they are often modeling a much bigger network and can have arbitrarily larger
stages and network size.  With varying assumptions on the latent variables and
their connectivity, there are many efficient algorithms that can perform either
approximate or exact inference on them.  Most applications in ML seem to follow
the Bayesian networks nomenclature since its context is more general.  We'll
stick with this framing since most of the sources will think about it this way.


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
2. Accept this new point (state), and transition to it with some probability calculated using
   the target distribution (or some function proportional to it).  Otherwise,
   stay at the current point (state).
3. Repeat steps 1 and 2, and periodically output the current point (state)

Many MCMC algorithms follow this general framework.  The key is ensuring
that the proposal and the acceptance probability define a Markov chain such
that the stationary distribution (i.e., steady state) is the same as your
target distribution.  See my previous post on `MCMC <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__ for more details.

Two additional complications.  The first complication is that your initial
state may be in some weird region that causes the algorithm to explore parts of
the state space that are low probability.  To solve this, you can perform
"burn-in" by starting the algorithm and throwing away a bunch of the initial
states to have a higher change to be in a more "normal" region of the state
space.  The other complication is that sequential samples will be correlated,
but ideally you want independent samples.  Thus (as specified in the steps
above), we only output the current state as a sample periodically to ensure
that the we have minimal correlation.  A well tuned MCMC algorithm will have
both a high acceptance rate and little correlation between samples.

`Hamiltonian Monte Carlo <https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo>`__  (HMC)
is a popular MCMC algorithm that has a high acceptance rate with low
correlation between samples.  It roughly transforms the target probability
distribution into a physics problem with `Hamiltonian dynamics <https://en.wikipedia.org/wiki/Hamiltonian_mechanics>`__.
Intuitively, the problem is similar to a frictionless puck moving along a 2D surface.
The position variables :math:`q` represent the state from our probability
distribution, and the momentum :math:`p` (equivalently velocity) are a set of
instrument variables to make the problem work.  For each proposal point, we
randomly pick a new momentum (and thus energy level of the system) and simulate
from our current point.  The end point is our new proposal point.

Simulating the associated differential equations of this physical system a
proposal point that both has a high acceptance rate and is "far away" (thus low
correlation).  In fact, the acceptance rate would be 100% if it not for the
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
variables, :math:`epsilon` is the step size of the discretized simulation, and
:math:`H := U(q) + K(p)` is the Hamiltonian, which (in this case) equals the
sum of potential energy :math:`U(q)` and the kinetic energy :math:`K(p)`.  The
potential energy is typically the negative logarithm of the target density up
to a constant :math:`f({\bf q})`, and the kinetic energy is usually defined as
independent zero-mean Gaussians with variances :math:`m_i`:

.. math::

   U({\bf q}) &= -log[f({\bf q})]  \\
   K({\bf p}) &= \sum_{i=1}^D \frac{p_i^2}{2m_i}  \\
   \tag{5}

A key fact is that the partial derivative of the Hamiltonian with respect to
the position or momentum results in the time derivative of the other one,
which are called *Hamilton's equations*:

.. math::

   \frac{\partial H}{\partial p} &= \frac{dq}{dt} \\
   \frac{\partial H}{\partial q} &= -\frac{dp}{dt} \\
   \tag{6} 

This result is used to derive Hamiltonian dynamics, but we'll also be using it momentarily.
Once we have a new proposal state :math:`(q^*, p^*)`, we accept the new state
according to this probability using a 
`Metropolis-Hasting <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`__ update:

.. math::

       A(q^*, p^*) = \min[1, \exp\big(-U(q^*) + U(q) -K(p^*)+K(p)\big)] \tag{7}

Langevin Monte Carlo
--------------------

Langevin Monte Carlo (LMC) [Radford2012]_ is a special case of HMC where we only
take a *single* step in the simulation to propose a new state (versus multiple
steps in a typical HMC algorithm).  It is sometimes referred to as the
Metropolis-Adjusted-Langevin algorithm (MALA), see [Teh2015]_ and references
for more details.  With some simplification, we will see that a new familiar
behavior emerges from this special case.

Suppose we define kinetic energy as :math:`K(p) = \frac{1}{2}\sum p_i^2`,
which is typical for a HMC formulation.  Next, we set our momentum :math:`p` as
a sample from a zero mean, unit variance Gaussian (still same as HMC). 
Finally, we run a single step of the leap frog to get new a new proposal state 
:math:`q^*` and :math:`p^*`.

We only need to focus on the position :math:`q` because we resample the
:math:`p` on each new proposal state and are only simulating one step so
:math:`p` gets reset anyways.  Starting from Equation 3:

.. math::

   q_i^* &= q_i(t) + \epsilon \frac{\partial H}{\partial p}(p(t+\epsilon/2))  \\
       &= q_i(t) + \epsilon \frac{\partial [U(q) + K(p)]}{\partial p}(p(t+\epsilon/2))  \\
       &= q_i(t) + \epsilon \frac{\partial [U(q) + \frac{1}{2}\sum p_i^2]}{\partial p}(p(t+\epsilon/2))  && \text{Per def. of kinetic energy} \\
       &= q_i(t) + \epsilon p|_{p=p(t+\epsilon/2)}  \\
       &= q_i(t) + \epsilon [p(t) - \frac{\epsilon}{2} \frac{\partial H}{\partial q_i}(q(t))] && \text{Eq. } 2 \\
       &= q_i(t) - \frac{\epsilon^2}{2} \frac{\partial H}{\partial q_i}(q(t)) + \epsilon p(t) \\
   \tag{8}

Equation 8 is known in physics as (one type of) Langevin Equation (see box for explanation),
thus the name Langevin Monte Carlo.

Now that we have a proposal state (:math:`q^*`), we can view the algorithm
as running a vanilla Metropolis-Hastings update where the proposal is coming
from a Gaussian with mean :math:`q_i(t) - \frac{\epsilon^2}{2} \frac{\partial H}{\partial q_i}(q(t))`
and variance :math:`\epsilon^2` corresponding to Equation 8.
By eliminating :math:`p` (and the associated :math:`p^*`, not shown here) from
the original HMC acceptance probability in Equation 7, we can derive the
following expression:

.. math::

   A(q^*) = \min\big[1, \frac{\exp(-U(q^*))}{\exp(-U(q))} 
        \Pi_{i=1}^d 
            \frac{\exp(-(q_i - q_i^* + (\epsilon^2 / 2) [\frac{\partial U}{\partial q_i}](q^*))^2 / 2\epsilon^2)}
            {\exp(-(q_i^* - q_i + (\epsilon^2 / 2) [\frac{\partial U}{\partial q_i}](q))^2 / 2\epsilon^2)}\big] \\
    \tag{9}

Even though LMC is derived from HMC, its properties are quite different.
The movement between states will be a combination of the :math:`\frac{\epsilon^2}{2} \frac{\partial H}{\partial q_i}(q(t))`
term and the :math:`\epsilon p(t)`.  Since :math:`\epsilon` is necessarily
small (otherwise your simulation will not be accurate), the former term
will be very small and the latter term will resemble a simple
Metropolis-Hastings random walk.  A big difference though is that LMC
has better scaling properties when increasing dimensions.  See [Radford2012]_
for more details.

Finally, we'll want to re-write equation 8 using different notation
to line up with our usual notation for stochastic gradient descent.
First, we'll use :math:`\theta` instead of :math:`q` to imply that
we're sampling from parameters of our model.  Next, we'll
rewrite the potential energy :math:`U(\theta)` as the likelihood times prior
(where :math:`x_i` are our observed data points):

.. math::

    U(\theta_t) &= -log[f(\theta_t)] \\
                &= -\log[p(\theta_t)] - \sum_{i=1}^N \log[p(x_i | \theta_t)] \\
    \tag{10}

Simplifying our Equation 8, we get:

.. math::

    
    \theta_{t+1} &= \theta_t - \frac{\epsilon_0^2}{2} \frac{\partial H}{\partial \theta} + \epsilon_0 p(t) \\
    \theta_{t+1} &= \theta_t - \frac{\epsilon_0^2}{2} \frac{\partial [U(\theta) + K(p)]}{\partial \theta} + \epsilon_0 p(t) \\
    \theta_{t+1} &= \theta_t- \frac{\epsilon_0^2}{2} \frac{\partial [-\log[p(\theta_t)] - \sum_{i=1}^N \log[p(x_i | \theta_t)]]}{\partial \theta} + \epsilon_0 p(t) && \text{Eq. } 10\\
    \theta_{t+1} - \theta_t &= \frac{\epsilon_0^2}{2} \big (\nabla \log[p(\theta_t)] + \sum_{i=1}^N \nabla \log[p(x_i | \theta_t)]]\big) + \epsilon_0 p(t) \\
    \theta_{t+1} - \theta_t &= \frac{\epsilon}{2} \big (\nabla \log[p(\theta_t)] + \sum_{i=1}^N \nabla \log[p(x_i | \theta_t)]]\big) + \sqrt{\epsilon} p(t) && \epsilon := \epsilon_0^2\\
    \Delta \theta_t &= \frac{\epsilon}{2} \big (\nabla \log[p(\theta_t)] + \sum_{i=1}^N \nabla \log[p(x_i | \theta_t)]]\big) + \varepsilon && \varepsilon \sim N(0, \epsilon) \\
    \tag{11}

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
   Equation A.2 is the same form as Equation 11 except a discretized version of
   it.  The only thing that the notation hides is that increments of the
   standard Weiner process :math:`W_t` are zero-mean Gaussians with variance
   equal to the time difference.  Once discretized with stepsize
   :math:`\epsilon`, this precisely equals our :math:`\varepsilon` sample from
   Equation 11.



Stochastic Gradient Descent and RMSProp
---------------------------------------

I'll only briefly cover stochastic gradient descent because I'm assuming most
readers will be very familiar with this algorithm.  
`Stochastic gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`__ (SGD)
is an iterative stochastic optimization of gradient descent.  The main difference
is that it uses a randomly selected subset of the data to estimate gradient at 
each step.  For a given statistical model with parameters :math:`\theta`,
log prior :math:`\log p(\theta)`, and log likelihood :math:`\sum_{i=1}^N \log[p(x_i | \theta_t)]]`
with observed data poits :math:`x_i`, we have:

.. math::

    \Delta \theta_t = \frac{\epsilon_t}{2} \big (\nabla \log[p(\theta_t)] 
    + \frac{N}{n} \sum_{i=1}^n \nabla \log[p(x_{ti} | \theta_t)]]\big) 
      \tag{12}

where :math:`\epsilon_t` is a sequence of step sizes, and each iteration :math:`t`
we have a subset of :math:`n` data points called a *mini-batch*
:math:`X_t = \{x_{t1}, \ldots, x_{tn}\}`.
By using an approximate gradient, over many iterations the entire dataset is used
and the noise in the estimated gradient averages out.  Additionally for large
datasets where the estimated gradient is accurate enough, this gives significant
computational savings versus using the whole dataset at each iteration.

Convergence to a local optimum is guaranteed with some mild assumptions combined
with a major requirement that the step size :math:`\epsilon_t` satisfies:

.. math::

   \sum_{t=1}^\infty \epsilon_t = \infty \hspace{50pt} \sum_{t=1}^\infty \epsilon_t^2 < \infty
   \tag{13}

Intuitively, the first constraint ensures that we make progress to reaching the
local optimum, while the second constraint ensures we don't just bounce around
that optimum.  A typical schedule to ensure that this is the case is using
a decayed polynomial:

.. math::

   \epsilon_t = a(b+t)^{-\gamma} \tag{14}

with :math:`\gamma \in (0.5, 1]`.

One of the issues with using vanilla SGD is that the gradients of the model
parameters (i.e. dimensions) may have wildly different variances.  For example,
one parameter may be smoothly descending at a constant rate while another may be
bouncing around quite a bit (especially with mini-batches).  To solve this, many
variations on SGD have been proposed that adjust the algorithm to account for the
variation in parameter gradients.  

`RMSProp <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp>`__
is a popular variant that is conceptually quite simple.  It adjusted the
learning rate *per parameter* to ensure that all of the learning rates are roughly
the same magnitude.  It does this by keeping a running average of the magnitudes
of recent gradients for parameter :math:`\theta` as :math:`v(\theta, t)`.
For :math:`j^{th}` parameter :math:`\theta^j` in iteration :math:`t`, we have:

.. math::

   v(\theta^j, t) := \gamma v(\theta^j, t-1) + (1-\gamma)(\nabla Q_i(\theta^j))^2 \tag{15}

where :math:`Q_i` is the loss function, and :math:`\gamma` is the smoothing
constant of the average with typical value set at `0.99`.  With :math:`v(\theta^j, t)`,
the update becomes:

.. math::

   \Delta \theta^j := - \frac{\epsilon_t}{\sqrt{v(\theta^j, t)}} \nabla Q_i(\theta^j) \tag{16}

From Equation 16, when you have large gradients (:math:`\nabla Q >1`), it scales
the learning rate down; while if you have large gradients (:math:`\nabla Q < 1`),
it scales the learning rate up.  If :math:`\nabla Q` is constant in each
parameter but with different magnitudes, it will update each parameter by the
learning rate :math:`\eta_t`, attempting to descend each dimension at the same
rate.  Empirically, these variations of SGD are necessary to make SGD practical
for a wide range of models.

Variational Inference and the Reparameterization Trick
------------------------------------------------------

I've written a lot about variational inference in my past posts so I'll
keep this section brief and only touch upon the relevant parts.
If you want more detail and intuition, check out my posts on 
`Semi-supervised learning with Variational Autoencoders <link://slug/semi-supervised-learning-with-variational-autoencoders>`__,
and `Variational Bayes and The Mean-Field Approximation <link://slug/variational-bayes-and-the-mean-field-approximation>`__.

As we discussed above, our goal is to find the posterior, :math:`p(\theta|X)`,
that tells us the distribution of the :math:`\theta` parameters Unfortunately,
this problem is intractable for all but the simplest problems. How can we 
overcome this problem? Approximation! 

We'll approximate :math:`p(\theta|X)` by another known distribution :math:`q(\theta|X; \phi)` 
parameterized by :math:`\phi` (and usually conditioned on :math:`X` but not
necessarily).  Importantly, :math:`q(\theta|X; \phi)` often also has some
simplifying assumptions about its relationships with other variables. 
For example, you might assume that they are all independent of each other
e.g., :math:`q(\theta|X;\phi) = \pi_{i=1}^n q_i(\theta_i|X;\phi_i)`.

The nice thing about this approximation is that we turned the intractable problem
into an optimization one where we just want to find the parameters :math:`\phi`
of :math:`q(\theta|X;\phi)` that best match our posterior :math:`p(\theta|X)`.
How well our approximation matches our posterior is both dependent on the
functional form of :math:`q` as well as our optimization procedure.

In terms of "best match", the standard way of measuring it is to use
`KL divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__.
Without going into the derivation (see my `previous post <semi-supervised-learning-with-variational-autoencoders>`__),
one arrive at the evidence lower bound (ELBO) for a single data point :math:`X`:

.. math::

  \log{p(X)} &\geq -E_q\big[\log\frac{q(\theta|X;\phi)}{p(\theta,X;\phi)}\big]  \\
             &= E_q\big[\log p(\theta,X) - \log q(\theta|X;\phi)\big] \\
             &= E_q\big[\log p(X|\theta) + \log p(\theta) - \log q(\theta|X;\phi)\big] \\
             &= E_q\big[\text{likelihood} + \text{prior} - \text{approx. posterior} \big] \\
              \tag{17}

The left hand side of Equation 17 is constant (with respect to the observed
data), so maximizing the right hand side achieves our desired goal.  It just so
happens this looks a lot like finding a 
`MAP <https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation>`__ with a
likelihood and prior term.  The difference is that we have an additional term
for our approximate posterior and we have to take the expectation with respect
to samples from our approximate posterior.  When using a SGD approach, we can
sample points from the :math:`q` distribution and use it to approximate the
expectation in Equation 17.  In many cases though, it's not obvious how to
sample from :math:`q` because you also need to backprop through it.  

In the case of 
`Variational Autoencoders <link://slug/variational-autoencoders>`__,
we define a Gaussian posterior :math:`q(z|X;\phi)` on the latent variables
:math:`z`. This approximate posterior is defined by a neural network with
weights :math:`\phi` that output a mean and variance representing the
parameters of the Gaussian.  We will want to sample from :math:`q` to
approximate the expectation in Equation 17, but also backprop through :math:`q`
to update the weights :math:`\phi` of the approximate posterior.
You can't directly backprop through it but you can reparameterize it by
using a standard normal distribution, starting from Equation 17 (using
:math:`z` instead of :math:`\theta`):

.. math::

        &E_{z\sim q}\big[\log p(X|z) + \log p(z) - \log q(z|X;\phi)\big] \\
        &= E_{\epsilon \sim \mathcal{N}(0, I)}\big[(\log p(X|z) + \log p(z) - \log q(z|X;\phi))\big|_{z=\mu_z(X) + \Sigma_z^{1/2}(X) * \epsilon}\big] \\
        &\approx (\log p(X|z) + \log p(z) - \log q(z|X;\phi))\big|_{z=\mu_z(X) + \Sigma_z^{1/2}(X) * \epsilon} \\
        \tag{18}

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
Monte Carlo (Equation 11) with Stochastic Gradient Descent (Equation 12)
given by:

.. math::

    \Delta \theta_t &= \frac{\epsilon_t}{2} \big (\nabla \log[p(\theta_t)] + \frac{N}{n} \sum_{i=1}^n \nabla \log[p(x_{ti} | \theta_t)]\big) + \varepsilon \\
    \varepsilon &\sim N(0, \epsilon_t)  \\
    \tag{19}

This results in an algorithm that is mechanically equivalent to SGD except with
some Gaussian noise added to each parameter update.  Importantly though, there
are several key decisions:

* :math:`\epsilon_t` decreases towards zero just as in SGD.
* Balance the Gaussian noise :math:`\varepsilon` variance with the step size
  :math:`\epsilon_t` as in LMC.
* Ignore the Metropolis-Hastings updates (Equation 9) using the fact that
  rejection rates asymptotically go to zero as :math:`\epsilon_t \to 0`. 

This algorithm has the advantage of SGLD of being able to work on large data
sets (because of the mini-batches) while still computing uncertainty
(using LMC-like estimates).  The avoidance of the Metropolis-Hastings update is
key so that an expensive evaluation of the whole dataset is not needed at each
iteration.

The intuition here is that in earlier iterations this will behave much like SGD
stepping towards a local maximum because the large gradient overcomes the
noise.  In later iterations though with a small :math:`\epsilon_t`, the noise
dominates and the gradient plays a much smaller role resulting in each
iteration bouncing around the local maxima via a random walk (with a bias
towards the local maximum from the gradient), and in between the two
extremes, the algorithm should vary smoothly.  Thus with carefully selected
hyperparameters, you can pretty closely sample from the posterior distribution
(more on this later).

What is not obvious though is that why this should give correct the correct
result.  It surely will be able to get close to a local maximum (similar to
SGD) but why would it give the correct uncertainty estimates without the
Metropolis-Hastings update step?  The next subsection explains this using the
reasoning from [Welling2011].

Correctness of SGLD 
-------------------

*Note:* [Teh2015]_ *has the hardcore proof of SGLD correctness versus a very
informal sketch presented in the original paper* ([Welling2011]_) *.  I'll mainly
stick to the original paper's presentation (mostly because the hardcore proof
is way beyond my comprehension), but will call out a couple of notable things.*

To setup this problem, let us first define several quantities.
First the true gradient of the log probability,
which is just the negative of the gradient our our usual loss function
(with no mini-batches):

.. math::

   g(\theta) = \nabla \log p(\theta) + \sum_{i=1}^N \nabla \log p(X_i|\theta) \tag{20}

Next, let's define another related quantity:

.. math::

   h_t(\theta) = \nabla \log p(\theta) + \frac{N}{n}\sum_{i=1}^n \nabla \log p(X_{ti}|\theta) - g(\theta) \tag{21}

Equation 21 is essentially the difference between our SGD update (with
mini-batch :math:`t`) and the true gradient update (with all the data).
Notice that an SGD update can be obtained by canceling the last term
with :math:`h_t(\theta) + g(\theta)`.

Importantly, :math:`h_t(\theta)` is a zero-mean random variable with
finite variance :math:`V(\theta)`.  Since we're subtracting the
true gradient, our mini-batches should net out to zero-mean.
Similarly, the variance comes from the fact that we're randomly selecting
mini-batches.  

With these quantities, we can rewrite Equation 19 as:

.. math::

    \Delta \theta_t &= \frac{\epsilon_t}{2} \big (g(\theta_t) + h_t(\theta_t) \big) + \varepsilon \\
    \varepsilon &\sim N(0, \epsilon_t)  \\
    \tag{22}

With the above setup, we'll show two statements:

1. **Transition**: When we have large :math:`t`, the state transition
   of Equation 19/22 will be the same as LMC, that is, have its equilibrium
   distribution be the posterior distribution.
2. **Convergence**: That there exists a subsequence of :math:`\theta_1,
   \theta_2, \ldots` that converges to the posterior distribution.

With these two shown, we can see that SGLD (for large :math:`t`) will
eventually get into a state where we can *theoretically* sample the posterior
distribution.  The paper makes a stronger argument that the subsequence
convergence implies convergence of the entire sequence but it's not clear to me
that it is the case.  At the end of this subsection, I'll also mention a theorem
from the rigorous proof ([Teh2015]_) that gives a practical result where this
may not matter.

**Transition**

We'll argue that Equation 19/22 converges to the same transition probability
as LMC and thus its equilibrium distribution will be the posterior.

First notice that Equation 19/22 is the same LMC (Equation 11) except for the
additional randomness due to the mini-batches: :math:`\frac{N}{n} \sum_{i=1}^n \nabla \log[p(x_{ti} | \theta_t)]`.
This term is multiplied by a :math:`\frac{\epsilon_t}{2}` factor where as
the standard deviation from the :math:`\varepsilon` term is :math:`\sqrt{\epsilon_t}`.
Thus as :math:`\epsilon_t \to 0`, the error from the mini-batch term vs. LMC
will vanish faster than the :math:`\varepsilon` term, converging to the LMC
proposal distribution (Equation 11).

Next, we observe that LMC is a special case of HMC.  HMC is actually a
discretization of a continuous time differential equation.  The discretization
introduces error in the calcluation, which is the only reason why we need a
Metropolis-Hastings update (see previous post on `HMC <link://slug/hamiltonian-monte-carlo>`__).
However as :math:`\epsilon_t \to 0`, this error becomes negligible converging
to the continuous time dynamics, implying a 100% acceptance rate.  Thus, there
is no need for an MH update for very small :math:`\epsilon_t`. 

In summary for the large :math:`t`, the :math:`t^{th}` iteration of Equation
19/22 effectively defines the LMC Markov chain transition whose equilibrium
distribution is the desired posterior.  This would be fine if we had a fixed
:math:`t` but we are actually shrinking :math:`t` towards 0, thus it
defines a non-stationary Markov Chain and so we still need to show the actual
sequence will convert to the posterior.

**Convergence**

We will show that there exists some sequence of samples :math:`\theta_{t=a_1},
\theta_{t=a_2}, \ldots` that converge to the posterior for some strictly
increasing sequence :math:`a_1, a_2, \ldots` (note: the sequence is not
sequential e.g., `a_{n+1}` is likely much bigger than :math:`a_{n+1}`).

First we fix a small :math:`\epsilon_0` such that :math:`0 < \epsilon_0 << 1`.
Assuming :math:`\{\epsilon_t\}` satisfy the decayed polynomial property from
Equation 14, there exists an increasing subsequence :math:`\{a_n \}` such that 
:math:`\sum_{t=a_n+1}^{a_{n+1}} \epsilon_t \to \epsilon_0` as :math:`n \to \infty`.
That is, we can split the sequence :math:`\{\epsilon_t\}` into non-overlapping
segments such that successive segment approaches :math:`\epsilon_0`.  This can
be easily constructed by continually extending the current run until you go
over :math:`\epsilon_0`.  Since :math:`\epsilon_t` is decreasing, and we are
guaranteed that the sequence doesn't converge (Equation 13), we can always
construct the next segment with a smaller error that the previous one.

For large :math:`n`, if we look at each segment, the total Gaussian noise
injected will be the sum of each of the Gaussian noise injections.  The
`variance of sums of independent Gaussians <https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables>`__ 
is just the sum of the variances so the total variance will be 
:math:`O(\epsilon_0)`.  Thus, the injected noise (standard deviation)
will be on the order of :math:`O(\sqrt{\epsilon})`.  Given this,
we will want to show that the variance from the mini-batch error is
dominated by the injected noise.

To start, since :math:`\epsilon_0 << 1`, we have 
:math:`||\theta_t-\theta_{t=a_n}|| << 1` for :math:`t \in (a_n, a_{n+1}]` 
since the updates from Equation 19/22 cannot stray too far from where it
started.  Assuming the gradients vary smoothly (a key assumption) then
we can see the total update without the noise from a segment 
:math:`t \in (a_n, a_{n+1}]` (using Equation 22 minus the noise :math:`\varepsilon`) is:

.. math::

   \sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2}\big(g(\theta_t) + h_t(\theta_t)\big)
   = \frac{\epsilon_0}{2} g(\theta_{t=a_n}) + O(\epsilon_0) + \sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2} h_t(\theta_t) \tag{23}

We see that the :math:`g(\cdot)` summation expands into the gradient at
:math:`\theta_{t=a_n}` plus an error term :math:`O(\epsilon_0)`.  This is
from our assumption of :math:`||\theta_t-\theta_{t=a_n}|| << 1` plus
the gradients varying smoothly (`Lipschitz contiuity <https://en.wikipedia.org/wiki/Lipschitz_continuity>`__),
which imply that the difference between successive gradients will be less than 1
(for an appropriately small :math:`\epsilon_0`).  Thus, the total error will
be :math:`\frac{\epsilon_t}{2} O(1) = O(\epsilon_0)` from our original
construction above.

Next, we deal with the :math:`h_t(\cdot)` in Equation 23.  Since we know
that :math:`\theta_t` did not vary much in our interval :math:`t \in (a_n, a_{n+1}]`
given our :math:`\epsilon << 1` assumption, we have :math:`h_t(\theta_t) = O(1)`
in our interval since our gradients vary smoothly.  Additionally each
:math:`h_t(\cdot)` will be a random variable which we can assume to be
independent, thus IID (doesn't change argument if they are randomly
partitioned which will only make the error smaller).  Plugging this into
:math:`\sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2} h_t(\theta_t)`, we
see the variance is :math:`O(\sum_{t=a_n+1}^{a_{n+1}} (\frac{\epsilon_t}{2})^2)`.
Putting this together in Equation 23, we get:

.. math::

   \sum_{t=a_n+1}^{a_{n+1}} \frac{\epsilon_t}{2}\big(g(\theta_t) + h_t(\theta_t)\big)
   &= \frac{\epsilon}{2} g(\theta_{t=a_n}) + O(\epsilon) + O\Big(\sqrt{\sum_{t=a_n+1}^{a_{n+1}} (\frac{\epsilon_t}{2})^2}\Big) \\
   &= \frac{\epsilon}{2} g(\theta_{t=a_n}) + O(\epsilon) \\
   \tag{24}

From Equation 24, we can see the total stochastic gradient over our segment is
just the exact gradient starting from :math:`\theta_{t=a_n}` with step size
:math:`\epsilon_0` plus a :math:`O(\epsilon_0)` error term.  But recall our 
injected noise was of order :math:`O(\sqrt{\epsilon_0})`, which in turn dominates
:math:`O(\epsilon_0)`.  Thus for small :math:`\epsilon_0`, our sequence
:math:`\theta_{t=a_1}, \theta_{t=a_2}, \ldots` will approximate LMC and
converge to the posterior as required.

Now the above argument showing that there exists a subsequence that samples
from the posterior isn't that useful because we don't know what that
subsequence is!  But [Teh2015]_ provides a much more rigorous treatment
of the subject showing a much more useful result in Theorem 7.  Without
going into all of the mathematical rigour, I'll present the basic idea 
(from what I can gather).

    **Theorem 1:** (Summary of Theorem 7 from [Teh2015]_)
    For a test function :math:`\varphi: \mathcal{R}^d \to \mathcal{R}`, the
    expectation of :math:`\varphi` with respect to the exact posterior
    distribution :math:`\pi` can be approximated by the weighted sum of
    :math:`m` SGLD samples :math:`\theta_0 \ldots \theta_{m-1}` that holds
    almost surely (given some assumptions):

    .. math::

        \lim_{m\to\infty} \frac{\epsilon_1 \varphi(\theta_0) + \ldots + \epsilon_m \varphi(\theta_{m-1})}{\sum_{t=1}^m \epsilon_t} = \int_{\mathcal{R}^d} \varphi(\theta)\pi(d\theta)
        \tag{25}

Theorem 1 gives us a more practical way to utilize the samples from SGLD.
We don't need to generate the exact samples that we would from LMC,
instead we can just the SGLD samples with their respective step sizes to
compute a weighted average for any actual quantity we would want (e.g.
expectation, variance, credible interval etc.).  According to Theorem 1,
this will converge to the exact quantity using the true posterior.
See [Teh2015]_ for more details (if you dare!).

Preconditioning
---------------

Practical Considerations
------------------------

* warmup
* thinning


Bayes by Backprop
=================

- Used in neural networks
- Still uses VI

Experiments
===========

Simple Gaussian Mixture
-----------------------

Stochastic Volatility Model
---------------------------

Conclusion
==========

References
==========
* Wikipedia:
* Previous posts: `Markov Chain Monte Carlo and the Metropolis Hastings Algorithm  <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__, `Hamiltonian Monte Carlo <hamiltonian-monte-carlo>`__ 

.. [Welling2011] Max Welling and Yee Whye Teh, "`Bayesian Learning via Stochastic Gradient Langevin Dynamics <https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf>`__", ICML 2011.
.. [Blundell2015] Blundell et. al, "`Weight Uncertainty in Neural Networks <https://arxiv.org/abs/1505.05424>`__", ICML 2015.
.. [Li] Li et. al, "`Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks <https://arxiv.org/abs/1512.07666>`__", AAAI 2016.
.. [Ma] Yi-An Ma, Tianqi Chen, Emily B. Fox, "`A Complete Recipe for Stochastic Gradient MCMC <https://arxiv.org/abs/1506.04696>`__", NIPS 2015.
.. [Radford2012] Radford M. Neal, "MCMC Using Hamiltonian dynamics", `arXiv:1206.1901 <https://arxiv.org/abs/1206.1901>`__, 2012.
.. [Teh2015] Teh et. al, "Consistency and fluctations for stochastic gradient Langevin dynamics", `arXiv:1409.0578 <https://arxiv.org/abs/1409.0578>`__, 2015.
