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
and `here <hamiltonian-monte-carlo>`__ if you want the nitty gritty details
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

`Hamiltonian Monte Carlo <https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo>`__ 
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
equations.  See my previous post on `HMC <https://en.wikipedia.org/wiki/Hamiltonian_mechanics>`__ for more details.

A common method for simulation of this physics problem uses the "leap frog" method
where we discretize time and simulate time step-by-step:

.. math::

   p_i(t+\epsilon/2) &= p_i(t) - \epsilon/2 \frac{\partial H}{\partial q_i}(q(t)) \tag{2}\\
   q_i(t+\epsilon) &= q_i(t) + \epsilon \frac{\partial H}{\partial p_i}(p(t+\epsilon/2)) \tag{3} \\
   p_i(t+\epsilon) &= p_i(t+\epsilon/2) - \epsilon/2 \frac{\partial H}{\partial q_i}(q(t+\epsilon)) \tag{4}

Where :math:`i` is the dimension index, :math:`q(t)` represent the position
variables at time :math:`t`, :math:`p(t)` similarly represent the momentum
variables, :math:`epsilon` is the step size of the discretized simulation, and
:math:`H := U(q) + K(p)` is the Hamiltonian, which (in this case) equals the
sum of potential energy :math:`U(q)` and the kinetic energy :math:`K(p)`.  The
potential energy is typically the negative logarithm of the target density up
to a constant (:math:`f({\bf q})`, and the kinetic energy is usually defined as
independent zero-mean Gaussians with variances :math:`m_i`:

.. math::

   U({\bf q}) &= -log[f({\bf q})]  \\
   K({\bf p}) &= \sum_{i=1}^D \frac{p_i^2}{2m_i}  \\
   \tag{5}

A key fact is that the partial derivative of the Hamiltonian with respect to
the position or momentum results in the time derivative of the other one:

.. math::

   \frac{\partial H}{\partial p} &= \frac{dq}{dt} \\
   \frac{\partial H}{\partial q} &= \frac{dp}{dt} \\
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
steps in a typical HMC algorithm).  With some simplification, we will see that
a new familiar behavior emerges from this special case.

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
       &= q_i(t) + \epsilon [p(t) - \epsilon/2 \frac{\partial H}{\partial q_i}(q(t))] && \text{Eq. } 2 \\
       &= q_i(t) + \frac{\epsilon^2}{2} \frac{\partial H}{\partial q_i}(q(t)) + \epsilon p(t) \\
   \tag{8}

Equation 8 is known in physics as (one type of) Langevin Equation (see box for explanation),
thus the name Langevin Monte Carlo.

.. admonition:: Langevin's Equation

   Langevin's equation

Now that we have a proposal state (:math:`q^*`), we can view the algorithm
as running a vanilla Metropolis-Hastings update where the proposal is coming
from a Gaussian with mean :math:`q_i(t) + \frac{\epsilon^2}{2} \frac{\partial H}{\partial q_i}(q(t))`
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
term and the math:`\epsilon p(t)`.  Since :math:`\epsilon` is necessarily
small (otherwise your simulation will not be accurate), the former term
will be very small and the latter term will resemble a simple
Metropolis-Hastings random walk.  The one difference is that LMC
has better scaling properties when increasing dimensions.  See [Radford2012]_
for more details.


Stochastic Gradient Descent and RMSProp
---------------------------------------

- SGD
- SGD guarantees
- RMSProp 

Variational Inference
---------------------

- VI, q-approx function
- ELBO
- Reparameterization trick

Stochastic Gradient Langevin Dynamics 
=====================================

- Explain intuition
- Proof of correctness

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
