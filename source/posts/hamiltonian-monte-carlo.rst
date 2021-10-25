.. title: Hamiltonian Monte Carlo
.. slug: hamiltonian-monte-carlo
.. date: 2021-09-11 20:47:05 UTC-04:00
.. tags: Hamiltonian, Monte Carlo, MCMC, Bayesian, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

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
`here <link://slug/tensors-tensors-tensors>`__,
`here <link://slug/manifolds>`__, and
`here <link://slug/hyperbolic-geometry-and-poincare-embeddings>`__), 
all of a sudden learning "advanced physics" doesn't seem so tough (especially
with all the amazing lectures and material online, it's actually easier than
ever)! Most of the material is based on [1] and [2], which I've found
area great sources for their respective areas.

This post is the culmination of many different rabbit holes (many much deeper
than I needed to go) where I'm going to try to explain HMC in simple and
intuitive terms to a satisfactory degree (that's the tag line of this blog
after all).  I'm going to begin by briefly motivating the topic by reviewing
MCMC and the Metroplis Hastings algorithm then move on to explaining
Hamiltonian dynamics (i.e., the "advanced physics"), and finally discuss HMC
with some toy experiments.

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
    

Background
==========

Markov Chain Monte Carlo
------------------------

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


Motivation
--------------------------------------

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

  **Figure 2: It's difficult to calibrate random walk MH algorithms**

From Figure 2, consider a bimodal distribution with a random walk MH algorithm.
If you start in one of the modes (left side) with a very tight proposal distribution (Proposal A), 
you may get "stuck" in that mode without visiting the other mode.
Theoretically, you'll eventually end up in the other mode but practically you
might not get there with a finite MCMC run.  
On the other hand, if you make the variance large (Proposal B) then in many
cases you'll end up in places where :math:`f(x')` is small, making the
acceptance rate from Equation 1 small.  There's no easy way around it and
finding the right variance will have to be tuned to your specific problem.

However, we've just been talking about random walk proposal distributions.
What if there was a better way?  Perhaps one where you can (theoretically)
get close to a 100% acceptance rate?  How about one where you don't need to throw
away any samples (Step 4 from MH algorithm above)?  Sounds too good to be true
doesn't it?  Yes, yes it is too good to be true, but we can *sort of* get there
with Hamiltonian Monte Carlo!  But first an explanation of Hamiltonian
Dynamics.

Hamiltonian Mechanics
=====================

Before we dive into Hamiltonian dynamics, let's do a quick review of high
school physics with Newton's second law of motion to understand how we can use
it to describe the motion of (macroscopic) objects.  Then we'll move onto
a more abstract method of describing these systems with Lagrangian mechanics.
Finally, we'll move on to Hamiltonian mechanics, which can be considered as a
modification of Lagrangian mechanics.  We'll see that these concepts are not
as scary as they sound as long as we remember some calculus and how to solve
relatively simple differential equations.

Classical Mechanics
-------------------

`Classical mechanics <https://en.wikipedia.org/wiki/Classical_mechanics>`__ 
(or Newtonian mechanics) is the physical theory that describes the motion
macroscopic objects like a ball, spaceship or even planetary bodies. 
I'll won't go much into detail on classical mechanics and assume
you are familiar with the basic concepts from a first course in physics.

One of the main tools we use to describe motion in classical mechanics
is Newton's second law of motion:

.. math::

    {\bf F_{net}} = m{\bf a(t)} = m\frac{d^2\bf x(t)}{dt^2} \tag{2}

Where :math:`\bf F_{net}` is the net force on an object, :math:`m` is the mass
of the object, :math:`\bf a(t)` is the acceleration, :math:`\bf x(t)` is the
position (with respect a reference), and **bold** quantities are vectors.

Notice that Equation 2 is a differential equation, where :math:`x(t)` describe
the equation of motion of the object over time.  In high school physics, you
may not have had to solve differential equations and were given equations to
solve for :math:`x(t)` assuming a constant force, but now that we know better,
we can directly solve for it.

I won't spend too much more time on this except to give a running example that
we'll use throughout the rest of this section.

.. admonition:: Example 1: A Simple Harmonic Oscillator using classical mechanics.

  .. figure:: /images/hmc_mass_spring.gif
    :height: 200px
    :alt: Simple Harmonic Oscillator
    :align: center
  
    **Figure 3: Simple Harmonic Oscillator (source: [3])**

  Consider a mass (:math:`m`) suspended from a spring in Figure 3, where
  :math:`k` is the force constant of the spring and positive :math:`x` is the
  downward direction with :math:`x=0` set at the spring's equilibrium.
  Using Newton's second law (Equation 2), we get the following differential equation:

  .. math::

    {\bf F_{net}} = -kx + mg = m{\bf a(t)} = m\frac{d^2\bf x(t)}{dt^2} \tag{3}

  Rearranging:

  .. math::

     \frac{d^2\bf x(t)}{dt^2} &= -\frac{k}{m}x(t) + g \\
                              &= -\frac{k}{m}(x(t) - x_0) && \text{rename }x_0 = g \\
                              &= -\frac{k}{m}x'(t)  && \text{define } x'(t) = x(t) - x_0 \\
     \tag{4}

  Here we are defining a new function :math:`x'(t)` that is shifted by :math:`-x_0`.
  This is basically the same as defining a new coordinate system shifted by
  :math:`-x_0` from our original one.
  Notice that :math:`\frac{d^2\bf x'(t)}{dt^2} = \frac{d^2\bf x(t)}{dt^2}`
  since the constant vanishes with the derivative.  And so we end up with the
  simplified differential equation:

  .. math::

    \frac{d^2\bf x'(t)}{dt^2} = -\frac{k}{m}x'(t) \tag{5}

  In this case, it's a second order differential equation with complex roots.
  I'll spare you solving it from scratch and just point you to this excellent
  `set of notes <https://tutorial.math.lamar.edu/Classes/DE/ComplexRoots.aspx>`__
  by Paul Dawkins.  However, we can also just see by observation that a solution
  is:

  .. math::

    x'(t) = Acos(\frac{k}{m}t + \phi) \tag{6}

  Given an initial position and its velocity, we can solve Equation 6 for the
  particular constants.

Example 1 gives the general idea of how to find the motion of an object:

1. Calculate the net forces.
2. Solve the (typically second order) differential equation from Equation 2 (Newton's second law).
3. Apply initial conditions (usually position and velocity) to find the constants.

It turns out this is not the only way to find the equation of motion.  The next section
gives us an alternative that is *sometimes* more convenient to use.

Lagrangian Mechanics
--------------------

Instead of using the classical formulation to solve the equation, we can use 
the Lagrangian method.  It starts out by defining this strange quantity
called the *Lagrangian*:

.. math::

    L(x(t), \frac{dx(t)}{dt}, t) = K - U = \text{Kinetic Energy} - \text{Potential Energy} \tag{7}

Where the Lagrangian is (typically) a function of the position :math:`x(t)`,
its velocity :math:`\frac{dx(t)}{dt}` and time :math:`t`.
It is kind of strange that we have a minus sign here and not a plus (which would give
the total energy).  We're going to show that we can use the Lagrangian to
arrive the same mathematical statement as Newton's second law by way of a
different method.  It's going to be a bit round about but we'll go through
several mathematical useful tools along the way (and will eventually lead us to
the Hamiltonian).

We'll start off by defining what is called the *action* that uses the Lagrangian:

.. math::
   
   S[x(t)] &= \int_{t_1}^{t_2} L(x(t),\frac{dx(t)}{dt}, t) dt \\
           &= \int_{t_1}^{t_2} L(x(t),x'(t), t) dt && \text{denote }  x'(t) := \frac{dx(t)}{dt} \\
   \tag{8}

The astute reader will notice that Equation 8 is a functional.  Moreover, it's precisely
the functional defined by the 
`Euler-Lagrange equation <https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation#Statement>`__.
For those who have not studied this topic, I'll give a brief overview here but 
direct you to my blog post on `the calculus of variations <link://slug/the-calculus-of-variations>`__
for more details.

Equation 8 is what is called a *functional*: a function :math:`S[x(t)]` of a function :math:`x(t)`,
where we use the square bracket to indicate a functional.  That is, if you plug in one function :math:`x_1(t)`
you get a scalar out; if you plug in another function :math:`x_t(t)`, you get another scalar out.  It's a mapping
from functions to scalars (as opposed to scalars to scalars in a normal single input function).

Equation 8 depends only on the function :math:`x(t)` (and it's derivative)
since :math:`t` gets integrated out.  Functionals have a lot of similarities to the traditional
functions we are used to in calculus, in particular they have the analogous concept of derivatives
called functional derivatives (denoted by :math:`\frac{\delta S}{\delta x}`).
One simple way to compute the functional derivative is to use the Euler-Lagrange equation:

.. math::

   \frac{\delta S[x]}{\delta x} 
   = \frac{\partial L}{\partial x} - \frac{d}{dt} \frac{\partial L}{\partial x'} \tag{9}

Here I'm dropping the parameters of :math:`L` and :math:`x` to make things a
bit more readable.  Equation 9 can be computed using our usual rules of
calculus since :math:`L` is just a multivariate function of :math:`t` (and not
a functional).  The proof of Equation 9 is pretty interesting but I'll refer
you to Chapter 6 of [2] if you're interested (which you can find online as a
sample chapter).

.. admonition:: Historical Remark

   As with a lot of mathematics, the Euler-Lagrange equation has its roots in physics.
   A young Lagrange at the age of 19 
   solved the `tautochrone problem <https://en.wikipedia.org/wiki/Tautochrone_curve>`__
   in 1755 developing many of the mathematics ideas described here.  He later
   sent it to Euler and they both developed the ideas further which led to
   Lagrangian mechanics.  Euler saw the potential in Lagrange's work and realized 
   that the method could extend beyond mechanics, so he worked with Lagrange to
   generalize it to apply to *any* functionals of that form, developing
   variational calculus in the process.

So why did we introduce all of these seemingly random expressions?  It turns
out that they are useful for the 
`principle of least action <https://en.wikipedia.org/wiki/Stationary-action_principle>`__:

    The path taken by the system between times :math:`t_1` and :math:`t_2` and
    configurations :math:`x_1` and :math:`x_2` is the one for which the *action* is stationary (no
    change) to first order.

where :math:`t_1` and :math:`t_2` are the initial and final times, and
:math:`x_1` and :math:`x_2` are the initial and final position.  It's sounds
fancy but what it's saying is that if you find a stationary function of Equation 8
(where the first functional derivative is zero) then it describes the motion of an object.
The classical mechanics result relies on quantum mechanics, which is beyond the
scope of this post (and my investigation on the subject).

However, if the principle of least action describe the motion then it should be equivalent
to the classical mechanics approach from the previous subsection -- and it indeed is equivalent!
We'll show this in the simple 1D case but it works in multiple dimensions and
with different coordinate basis as well.  Starting with a general Lagrangian (Equation 7)
for an object:

.. math::

    L(x(t), x'(t), t) = K - U = \frac{1}{2}mx'^2(t) - U(x(t)) \tag{10}

Here we're using the standard kinetic energy formula (:math:`K=\frac{1}{2}mv^2`, where velocity :math:`v=x'(t)`) and a 
generalized potential function :math:`-U(x(t))` that depends on the object's
position such as gravity.  Plugging :math:`L` into the Euler-Lagrange (Equation
8) and setting to zero to find the stationary point, we get:

.. math::

   \frac{\partial L}{\partial x} - \frac{d}{dt} \frac{\partial L}{\partial x'} &= 0 \\ 
   \frac{\partial L}{\partial x} &= \frac{d}{dt} \frac{\partial L}{\partial x'} \\ 
   \frac{\partial [\frac{1}{2}mx'^2(t) - U(x(t))]}{\partial x} &= \frac{d}{dt} \frac{\partial [\frac{1}{2}mx'^2(t) - U(x(t))]}{\partial x'} \\ 
   -\frac{\partial - U(x(t))}{\partial x} &= \frac{d[mx'(t)]}{dt} \\ 
   -\frac{\partial U(x(t))}{\partial x} &= mx''(t) \\ 
   F = ma(t) && a(t) = \frac{d^2x}{dx^2} \text{ and F}= -\frac{\partial U(x(t))}{\partial x} \\ 
   \tag{11}

So we can see that we end up with Newton's second law of motion as we expected.
The negative sign comes in because if we decrease the potential (change in
potential is negative), we're moving in the direction of the potential field,
thus we have a positive force.  

So we went through all of that to derive the same equation?  Pretty much, but in
certain cases the Lagrangian is easier to formulate and solve than the
classical approach (although not in the simple example below).  Additionally,
it is going to be useful to help us derive the Hamiltonian.

.. admonition:: Example 2: A Simple Harmonic Oscillator using Lagrangian mechanics.

    Using the same problem in Example 1, let's solve it using the Lagrangian.
    We can define the Lagrangian as (omitting the parameters for cleanliness):

    .. math::

        L = K - U = \frac{1}{2}mx'^2 - (mgx + \frac{1}{2}kx^2) \tag{12}

    where each term represents the velocity, gravitational potential and
    elastic potential of the spring respectively.  Using the Euler-Lagrange
    equation (and setting it to 0):
   
    .. math:: 

        \frac{\partial L}{\partial x} &= \frac{d}{dt} \frac{\partial L}{\partial x'} \\
        \frac{\partial [\frac{1}{2}mx'^2 - (mgx + \frac{1}{2}kx^2)]}{\partial x} &= \frac{d}{dt} \frac{\partial [\frac{1}{2}mx'^2 - (mgx + \frac{1}{2}kx^2)]}{\partial x'} \\
        -mg - kx &= mx'' \\
        -g - \frac{k}{m}x &= x''  \\
        \frac{d^2x}{dt^2} &= -\frac{k}{m}(x - x_0) && \text{rename } x_0 = g \\
        \tag{13}

    And we see we end up with the same second order differential equation as
    Equation 4, which yields the same solution :math:`x'(t) = Acos(\frac{k}{m}t + \phi)`.
    As you can see, we didn't really gain anything by using the Lagrangian but 
    often times in multiple dimensions, potentially with a different coordinate
    basis, the Lagrangian method is easier to use.


One last note before we move on to the next section.  It turns out the
Euler-Lagrange from Equation 9 is agnostic to the coordinate system we are using.
In other words, for another coordinate system :math:`q_i:= q_i(x_1,\ldots,x_N;t)`
(with the appropriate inverse mapping :math:`x_i:= x_i(q_1,\ldots,q_N;t)`),
then the Euler-Lagrange equation works with the new coordinate system as well
(at the stationary point):

.. math::

   \frac{d}{dt} \frac{\partial L}{\partial q'_m} = \frac{\partial L}{\partial q_m} && 1 \leq m \leq N \\
   \tag{14}

From here on out instead of assuming Cartesian coordinates (denoted with
:math:`x`'s), we'll be using the generic :math:`q` to denote position
with its corresponding first (:math:`q'`) and second derivatives (:math:`q''`)
for velocity and acceleration, respectively.

Hamiltonian Mechanics
---------------------

We're slowly making our way towards HMC and we're almost there!  Finally,
let's discuss how we can solve the equation of motion using Hamiltonian mechanics.
We first start off with another esoteric quantity:

.. math::

    E := \big(\sum_{i=1}^N \frac{\partial L}{\partial q'_i} q'_i \big) - L \tag{15}

where we have potentially :math:`N` particles and/or coordinates.  The symbol
:math:`E` is used because *usually* Equation 15 is the total energy of the
system.  Let's show that in 1D using the fact that
:math:`L=K-U=\frac{1}{2}mq'^2 - U(q)` for potential energy :math:`U(q)`:

.. math::

   E &:= \frac{\partial L}{\partial q'} q' - L \\
     &= \frac{\partial (\frac{1}{2}mq'^2 - U(q))}{\partial q'} q' - L \\
     &= mq' \cdot q'_i - L \\
     &= 2K - (K - U) \\
     &= K + U \\
     \tag{16}

where we can see that it's the kinetic energy *plus* the potential energy of
the system.  If the coordinate system you are using are Cartesian, then it is
always the total energy.  Otherwise, you have to ensure the change of basis
does not have a time dependence or else there's not guarantee.  See 15.1 from
[2] for more details.

Now we're almost at the Hamiltonian with Equation 15 but we want to do a
variable substitution:

.. math::

    p := \


Hamiltonian Monte Carlo
=======================




Experiments
===========

Conclusion
==========


Further Reading
===============

* Previous posts: `Markov Chain Monte Carlo Methods, Rejection Sampling and the Metropolis-Hastings Algorithm <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__, `The Calculus of Variations <link://slug/the-calculus-of-variations>`__
* Wikipedia: `Metropolis-Hastings Algorithm <https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm>`__, 
  `Classical Mechanics <https://en.wikipedia.org/wiki/Classical_mechanics>`__,
  `Lagrangian Mechanics <https://en.wikipedia.org/wiki/Lagrangian_mechanics>`__,
  `Hamiltonian Mechanics <https://en.wikipedia.org/wiki/Hamiltonian_mechanics>`__
* [1] Radford M. Neal, MCMC Using Hamiltonian dynamics, `arXiv:1206.1901 <https://arxiv.org/abs/1206.1901>`__, 2012.
* [2] David Morin, `Introduction to Classical Mechanics <https://scholar.harvard.edu/david-morin/classical-mechanics>`__, 2008.
* [3] `HyperPhysics <http://hyperphysics.phy-astr.gsu.edu/hbase/shm2.html>`__
