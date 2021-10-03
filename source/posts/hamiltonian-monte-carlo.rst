.. title: Hamiltonian Monte Carlo
.. slug: hamiltonian-monte-carlo
.. date: 2021-09-11 20:47:05 UTC-04:00
.. tags: Hamiltonian, Monte Carlo, MCMC, Bayesian, mathjax
.. category: 
.. link: 
.. description: 
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

.. |hr| raw:: html

   <hr>

Here's a topic that I thought I was too complex that I would never
come back to it.  When I first started learning about Bayesian methods, I was
aware enough that I should know something about MCMC since that's the backbone
of most Bayesian analysis; so I learned something about it
(see my `previous post <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__).
But I didn't dare try to go to the depths of trying to learn about the
notorious Hamiltonian Monte Carlo (HMC). Even though it is **the** standard algorithm
that is used to solve Bayesian inference, it always seemed too daunting because
it required "advanced physics" to understand.  As usual, things only seem hard
because you don't know them yet.  After having some time to digest MCMC
methods, getting comfortable learning more maths (see 
`here <link://slug/tensors-tensors-tensors/>`__,
`here <link://slug/manifolds/>`__, and
`here <link://slug/hyperbolic-geometry-and-poincare-embeddings/>`__), 
all of a sudden learning "advanced physics" doesn't seem so tough (especially
with all the amazing lectures and material online, it's actually easier than
ever)!

This post is the culmination of many different rabbit holes (many much deeper
than I needed to go) where I'm going to try to explain HMC in simple and
intuitive terms to a satisfactory degree (that's the tag line of this blog
after all).  I'm going to begin by briefly motivating the topic by reviewing
MCMC and the Metroplis Hastings algorithm then move on to explaining
Hamiltonian dynamics (i.e., the "advanced physics"), and finally discuss HMC
with some toy experiments.

.. TEASER_END

|h2| Background: Markov Chain Monte Carlo |h2e|

This section is going to give a brief overview of MCMC and the
Metropolis-Hastings algorithm.  For a more detailed treatment, see my 
`previous post <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__,
which goes much more in depth.  I'm only going to review some of the main
relevant ideas in this section.

Markov Chain Monte Carlo (MCMC) are a class of algorithms that use Markov Chains to
sample from a particular probability distribution ("Monte Carlo").  The idea is that
you traverse states in a Markov Chain so that (assuming you constructed it correctly)
it approximates your target distribution.

.. figure:: /images/mcmc.png
  :height: 270px
  :alt: Visualization of a Markov Chain Monte Carlo
  :align: center

  **Figure 1: Visualization of a Markov Chain Monte Carlo.**

Figure 1 shows a crude visualization of the idea.  The "states" of the Markov Chain
are the support of your probability distribution (the figure only shows
states with discrete values for simplicity but they can also be continuous).
The goal is to construct a Markov Chain such that randomly traversing the
states your target distribution.

One of the earliest algorithms to accomplish this is called the `Metropolis-Hastings Algorithm <https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm>`__.
This algorithm is nice because you don't need the actual probability
distribution, call it :math:`p(x)`, but rather only a function :math:`f(x)
\propto p(x)`.  Assuming that the state space of the Markov Chain is the
support of your target probability distribution, the algorithm gives a method
to select the next state to traverse.  It does this by introducing two new
distributions: a *proposal distribution* :math:`g(x)` and an *acceptance
distribution* :math:`A(x)`.  The proposal distribution only needs to have the
same support as your target distribution, although it's much more efficient if
it has a similar shape.  The acceptance distribution is defined as:

.. math::
    A(x \rightarrow x') = min(1, \frac{f(x')g(x' \rightarrow x)}{f(x)g(x \rightarrow x')}) \tag{1}

with :math:`x'` being the newly proposed state sampled from :math:`g(x)`.  
The :math:`x \rightarrow x'` (and vice versa) symbol means that the
proposal distribution is conditioned on the current state i.e., :math:`x' | x`.
The idea is that the proposal distribution will change depending on the current
state.  A common choice is a normal distribution centered on :math:`x` with
a variance dependent on the problem.

The algorithm can be summarized as such:

1. Initialize the initial state by picking a random :math:`x`.
2. Find new :math:`x'` according to :math:`g(x \rightarrow x')`.
3. Accept :math:`x'` with uniform probability according to :math:`A(x \rightarrow x')`.  If accepted transition to :math:`x'`, otherwise stay in state :math:`x`.
4. Go to step 2, :math:`T` times.
5. Save state :math:`x` as a sample, go to step 2 to sample another point.

Notice step 4 where we throw away a bunch of samples before we return one.
This is because typically sequential samples will be correlated, which is the
opposite of what we want.  So we throw away a bunch of samples in hopes that
the sample we pick is sufficiently independent.  Theoretically as we approach
an infinite number of samples this doesn't make a difference but practically
we need it in order to generate random samples.

To make MH efficient, you want your proposal distribution to be accepting with
a high probability, otherwise you get stuck in the same state and it takes a
very long time for the algorithm to converge.  This means you want 
:math:`g(x \rightarrow x') \approx f(x')` (and vice versa).  If they are
approximately equal, then the fraction in Equation 1 is approximately 1. 
But this isn't so easy to do because if you could sample from the original
distribution then why would you need MCMC in the first place?  We'll see
how we can get pretty close though later on.


|h2| Motivation for Hamiltonian Monte Carlo |h2e|

Let's take a look at the basic case of using a normal distribution as our
proposal distribution (in 1D).  We can see that 
:math:`g(x \rightarrow x') = g(x' \rightarrow x)` since it is symmetric.
In other words, the probability of jumping from :math:`x` to :math:`x'` 
(with :math:`g` centered on :math:`x`) is the same as jumping from
:math:`x'` to :math:`x` (with :math:`g` centered on :math:`x'`).  So
the fraction in Equation 1 then becomes simply :math:`\frac{f(x')}{f(x)}`.
This implies that you're more than likely to stick around in state :math:`x`
if it has a high density, and unlikely to move to state :math:`x'` if it has
low density (and vice versa).

This method is typically called the "random walk" Metropolis-Hastings because
you're randomly selecting a point from your current location.  It works but
it's not without its problems.  The main issue is that it doesn't very
efficiently explore the state space.  Figure 2 shows a visualization of this
idea.

.. figure:: /images/hmc_motivation.png
  :height: 270px
  :alt: Bimodal distribution
  :align: center

  **Figure 2: Difficult to reach other modes with a random walk MH algorithm.**

From Figure 2, consider a bimodal distribution with a random walk MH algorithm.
If you start in one of the modes (left side), you may get "stuck" in that mode
without visiting the other mode, especially if your proposal distribution has a
small variance.  Theoretically, you'll eventually end up in the other mode but
practically you might not get there with the finite MCMC run.
On the other hand, if you make the variance large then in many cases you'll end
up in places where :math:`f(x')` is small, making the acceptance rate from
Equation 1 small.  There's no easy way around it and finding the right variance
will have to be tuned to your specific problem.

However, we've just been talking about random walk proposal distributions.
What if there was a better way?  Perhaps one where you can theoretically
approach a 100% acceptance rate?  How about one where you don't need to throw
away any samples (Step 4 from MH algorithm above)?  Sounds too good to be
doesn't it?  Yes it does but we can sort of get there with Hamiltonian Monte
Carlo where we can usually do much better random walk MH.  But first an
explanation of Hamiltonian Dynamics.

|h2| Hamiltonian Dynamics |h2e|

|h2| Hamiltonian Monte Carlo |h2e|

|h2| Experiments |h2e|

|h2| Conclusion |h2e|


|h2| Further Reading |h2e|

* Previous posts: `Markov Chain Monte Carlo Methods, Rejection Sampling and the Metropolis-Hastings Algorithm <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__, 
* Wikipedia: `Metropolis-Hastings Algorithm <https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm>`__
* [1] Radford M. Neal, MCMC Using Hamiltonian dynamics, `arXiv:1206.1901 <https://arxiv.org/abs/1206.1901>`__, 2012.
