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

Three important conditions that are required for a Markov chain to be used for MCMC:

1. **Irreducible**: we must be able to reach any one state from any other state
   eventually (i.e. the expected number of steps is finite)
2. **Aperiodic**: the system never returns to the same state with a fixed
   period (e.g. not returning to start "sunny" deterministically every 5
   steps)
3. **Reversible**: a Markov chain is called `reversible <https://en.wikipedia.org/wiki/Detailed_balance#Reversible_Markov_chains>`__
   if the Markov chain has a stationary distribution :math:`\pi` such that
   :math:`\pi_iP_{ij} = \pi_jP{ji}` where :math:`P_ij` is the transition
   probability from state :math:`i` to :math:`j` and :math:`\pi_i` and
   :math:`\pi_j` are the equilibrium probabilities for their respective states.

The first two properties define a Markov chain which is `ergodic <https://nlp.stanford.edu/IR-book/html/htmledition/definition-1.html>`__,
which implies that a that there is a steady state distribution.
The third property is used to derive the MCMC algorithm defined below.
See my `previous post <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__,
for a derivation.

One of the earliest MCMC algorithms is called the `Metropolis-Hastings Algorithm <https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm>`__.
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
    A(x \rightarrow y) = min(1, \frac{f(y)g(y \rightarrow x)}{f(x)g(x \rightarrow y)}) \tag{1}

with :math:`y` being the newly proposed state sampled from :math:`g(x)`.  
The :math:`x \rightarrow y` (and vice versa) symbol means that the
proposal distribution is conditioned on the current state i.e., :math:`y | x`.
The idea is that the proposal distribution will change depending on the current
state.  A common choice is a normal distribution centered on :math:`x` with
a variance dependent on the problem.

The algorithm can be summarized as such:

1. Initialize the initial state by picking a random :math:`x`.
2. Find new :math:`y` according to :math:`g(x \rightarrow y)`.
3. Accept :math:`y` with uniform probability according to :math:`A(x \rightarrow y)`.  If accepted transition to :math:`y`, otherwise stay in state :math:`x`.
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

Note that I use the notation :math:`x'(t) := \frac{dx}{dt}` to always represent
the time derivative of the function :math:`x(t)` (or later on :math:`p` and
:math:`q`).  Most physics sources use the "dot" (:math:`\dot{x}(t)`) notation to
represent time derivatives but I'll use the apostrophe because I think it's probably 
more familiar to non-physics readers.

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
                              &= -\frac{k}{m}y(t)  && \text{define } y(t) = x(t) - x_0 \\
     \tag{4}

  Here we are defining a new function :math:`y(t)` that is shifted by :math:`-x_0`.
  This is basically the same as defining a new coordinate system shifted by
  :math:`-x_0` from our original one.
  Notice that :math:`\frac{d^2\bf y(t)}{dt^2} = \frac{d^2\bf x(t)}{dt^2}`
  since the constant vanishes with the derivative.  And so we end up with the
  simplified differential equation:

  .. math::

    \frac{d^2\bf y(t)}{dt^2} = -\frac{k}{m}y(t) \tag{5}

  In this case, it's a second order differential equation with complex roots.
  I'll spare you solving it from scratch and just point you to this excellent
  `set of notes <https://tutorial.math.lamar.edu/Classes/DE/ComplexRoots.aspx>`__
  by Paul Dawkins.  However, we can also just see by observation that a solution
  is:

  .. math::

    y(t) = Acos(\frac{k}{m}t + \phi) \tag{6}

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
called the *Lagrangian* [1]_:

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

        L = K - U = \frac{1}{2}mx'^2 - (-mgx + \frac{1}{2}kx^2) \tag{12}

    where each term represents the velocity, gravitational potential and
    elastic potential of the spring respectively.  Recall :math:`x=0` is defined
    to be where the spring is at rest and positive :math:`x` is the downward
    direction.  Thus, the gravitational potential is negative of the :math:`x`
    direction while the spring has potential with any deviation from :math:`x=0`.

    Using the Euler-Lagrange equation (and setting it to 0):
   
    .. math:: 

        \frac{\partial L}{\partial x} &= \frac{d}{dt} \frac{\partial L}{\partial x'} \\
        \frac{\partial [\frac{1}{2}mx'^2 - (-mgx + \frac{1}{2}kx^2)]}{\partial x} &= \frac{d}{dt} \frac{\partial [\frac{1}{2}mx'^2 - (-mgx + \frac{1}{2}kx^2)]}{\partial x'} \\
        mg - kx &= mx'' \\
        g - \frac{k}{m}x &= x''  \\
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

The Hamiltonian and Hamilton's Equations
----------------------------------------

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
variable substitution by getting rid of :math:`q'` and replacing it with
something called the *generalized momentum*:

.. math::

    p := \frac{\partial L}{\partial q'} \tag{17}

This is *sometimes* the same as the usual linear momentum (usually denoted by :math:`p`)
you learn about in a first physics class.  Assuming we have the usual equation for kinetic
energy with Cartesian coordinates:

.. math::

    p &:= \frac{\partial L}{\partial q'} \\
      &= \frac{\partial (\frac{1}{2}mq'^2 - U(q))}{\partial q'}
      &= mq'    && \text{linear momentum}\\
    \tag{18}

However, for example, if you are dealing with angular kinetic energy (such as a
swinging pendulum) and using those coordinates then you'll end up with 
`angular momentum <https://en.wikipedia.org/wiki/Angular_momentum>`__ instead.
In any case, all we need to know is Equation 17.  Substituting it into our
(often) total energy equation (Equation 15) and re-writing in terms of only
:math:`q` and :math:`p` (no explicit :math:`q'`), we get the Hamiltonian:

.. math::

    H({\bf q, p}) &= \big(\sum_{i=1}^N \frac{\partial L}{\partial q'_i} q'_i \big) - L  && \text{definition of } E \\
            &= \big(\sum_{i=1}^N p_i q'_i(q, p_i) \big) - L({\bf q, q'(q, p)})  && p_i := \frac{\partial L}{\partial q'_i}\\
    \tag{19}

where I've used bold to indicate vector quantities.  Notice that we didn't
explicitly eliminate :math:`q'_i`, we just wrote it as a function of :math:`q`
and :math:`p`.  

The :math:`2n` dimensional coordinates :math:`({\bf q, p})` are called the
*phase space coordinates* (also known as canonical coordinates).  Intuitively,
we can just think of this as the position (:math:`x`) and linear momentum
(:math:`mv = mx'`), which is what you would expect if you were asked for the
current state of a system (alternatively you could use velocity instead of
momentum).  However, as we'll see later, phase space coordinates have
certain nice properties that we'll utilize when trying to perform MCMC.

Now Equation 19 by itself maybe isn't that interesting but let's see what happens
when we analyze how it changes with respect to its inputs :math:`q` and :math:`p`
(in 1D to keep things cleaner).  Starting with :math:`p`:

.. math::

   \frac{\partial H}{\partial p} &= \frac{\partial (p q'(q, p))}{\partial p}  - \frac{\partial L(q, q'(q, p))}{\partial p} \\
                                 &= [q'(q, p) + p\frac{\partial (q'(q, p))}{\partial p}] 
                                    - \frac{\partial L(q, q'(q, p))}{\partial q'} \frac{\partial q'(q, p)}{\partial p} \\
                                 &= [q'(q, p) + p\frac{\partial q'(q, p)}{\partial p}] 
                                    - p \frac{\partial q'(q, p)}{\partial p} && p := \frac{\partial L}{\partial q'} \\
                                 &= q'(q, p) = q'
                                \tag{20} 

Now isn't that nice?  The partial derivative with respect to the generalized
momentum of the Hamiltonian simplifies to the velocity.  Let's see what happens
when we take it with respect to the position :math:`q`:

.. math::

   \frac{\partial H}{\partial q} &= \frac{\partial (p q'(q, p))}{\partial q}  - \frac{\partial L(q, q'(q, p))}{\partial q} \\
                                 &= p\frac{\partial q'(q, p)}{\partial q}  - 
                                    [\frac{\partial L(q, q')}{\partial q}  
                                     + \frac{\partial L(q, q')}{\partial q'} \frac{\partial q'(q, p)}{\partial q} ]
                                    && \text{See remark below} \\
                                 &= p\frac{\partial q'(q, p)}{\partial q}  
                                    - [\frac{d}{dt}\big( \frac{\partial L(q, q')}{\partial q'} \big) 
                                     + \frac{\partial L(q, q')}{\partial q'} \frac{\partial q'(q, p)}{\partial q} ]
                                    && \text{Euler-Lagrange equation} \frac{d}{dt}\big(\frac{\partial L}{\partial q'}\big) = \frac{\partial L}{\partial q} \\
                                 &= p\frac{\partial q'(q, p)}{\partial q}  
                                    - [\frac{dp}{dt} + p \frac{\partial q'(q, p)}{\partial q}]
                                    && p := \frac{\partial L}{\partial q'} \\
                                 &= -p'
                                \tag{21}

Similarly, we get a (sort of) symmetrical result where the partial derivative
with respect to the position is the negative first time derivative of the
generalized momentum. Equations 20 and 21 are called *Hamilton's equations*,
which will allow us to compute the equation of motion as we did in the previous
two methods.  The next example shows this in more detail.

.. admonition:: Explanation of :math:`\frac{\partial L(q, q'(q, p))}{\partial q} = \frac{\partial L(q, q')}{\partial q} + \frac{\partial L(q, q')}{\partial q'} \frac{\partial q'(q, p)}{\partial q}`

    This expression is *partially* (get it?) confusing because of the notation and partially confusing because
    it's not typically seen when discussing the chain rule for partial differentiation.  Notice that the LHS looks
    *almost* identical to the first term in the RHS.  The difference being that
    :math:`q'(q, p)` is a function of :math:`q` on the LHS, while on the RHS it's constant with respect to :math:`q`.
    To see that, let's re-write the LHS using some dummy functions.

    Define :math:`f(q) = q` and :math:`g(q, p) = q'(q, p)`, and then substitute into the LHS and apply the 
    `chain rule for partial differentiation <https://tutorial.math.lamar.edu/classes/calciii/chainrule.aspx>`__:

    .. math::

        \frac{\partial L(f(q), g(q, p))}{\partial q} &= 
            \frac{\partial L(f(q), g)}{\partial f}\Big|_{g=q'(q, p)}\frac{df(q)}{dq}
            + \frac{\partial L(f(q), g(q, p))}{\partial g}\frac{\partial g(q, p)}{\partial q} \\
            &= \frac{\partial L(q, g)}{\partial q}\Big|_{g=q'(q, p)}(1)
            + \frac{\partial L(q, g)}{\partial g}\frac{\partial g(q, p)}{\partial q} \\
            &= \frac{\partial L(q, q')}{\partial q}
            + \frac{\partial L(q, q')}{\partial q'}\frac{\partial q'(q, p)}{\partial q} \\
        \tag{22}

    As you can see the first term on the RHS has a "constant" :math:`q'` from
    the partial differentiation of :math:`f(q) = q`.  The notation seems a bit messy,
    I did a double take when I first saw it, but hopefully this makes it clear as mud.
   

.. admonition:: Example 3: A Simple Harmonic Oscillator using Hamiltonian mechanics.

    Using the same problem in Example 1 and 2, let's solve it using Hamiltonian
    mechanics.  We start by writing the Lagrangian (repeating Equation 12):

    .. math::

        L = K - U = \frac{1}{2}mx'^2 - (-mgx + \frac{1}{2}kx^2)

    Next, calculate the generalized momentum (Equation 17):

    .. math::

        p &:= \frac{\partial L}{\partial x'} \\
          &= mx' \\ \tag{23}

    Which turns out to just be the linear momentum.  Note, we'll
    be using :math:`x` instead of :math:`q` in this example since
    we'll be using standard cartesian coordinates.  
    
    From Equation 23, solve for the velocities (:math:`x'`) so we can re-write
    in terms of momentum, we get:

    .. math::

        p &= mx' \\
        x' &= \frac{p}{m} \\ tag{24}

    Write down the Hamiltonian (Equation 19) in terms of its phase
    space coordinates :math:`(x, p)`, eliminating all velocities
    using Equation 24:

    .. math::

        H({\bf x, p}) &= p x'(x, p) - L({\bf x, x'(x,p)}) \\
                      &= p \frac{p}{m} - [\frac{1}{2}mx'^2 - (-mgx + \frac{1}{2}kx^2)] \\
                      &= \frac{p^2}{m} - [\frac{1}{2}m(\frac{p}{m})^2 - (-mgx + \frac{1}{2}kx^2)] \\
                      &= \frac{p^2}{2m} - mgx + \frac{1}{2}kx^2 \\
        \tag{25}

    Write down Hamilton's equation (Equation 20 and 21):

    .. math::
    
        \frac{\partial H}{\partial x} &= -p' \\
        -mg + kx &= -p'  \\
        \frac{dp}{dt} &= -kx + mg \tag{26} \\
        \\
        \frac{\partial H}{\partial p} &= x' \\
        \frac{p}{m} &= x'  \\
        \frac{dx}{dt} &= \frac{p}{m} \tag{27}

    Finally, we just need to solve these differential equations for :math:`x(t)`.
    In general, this involves eliminating :math:`p` in favor of :math:`q'`. 
    In this case it's quite simple.  Notice that Equation 26 is exactly
    Newton's second law (where :math:`\frac{dp}{dt} = \frac{mx'}{dt} = ma`) and
    mirrors Equation 4, while Equation 27 is just the definition of velocity
    (where :math:`p=ma`).  As a result, we'll end up with exactly the same
    solution for :math:`x(t)` as the previous examples.

Properties of Hamiltonian Mechanics
-----------------------------------

After going through example 3, you may wonder what was the point of all of this
manipulation?  We essentially just ended with Newton's second law, which
required an even more round about way via writing the Lagrangian, Hamiltonian,
Hamilton's equations and then essentially converting back to where we started.
These are all very good observations and the simple examples shown so far don't
do Hamiltonian mechanics justice.  One typically does not use the
Hamiltonian method for standard mechanics problems involving a small number of
particles.  It really starts to shine when using it for analysis with a large
number of particles (e.g. thermodynamics) or with no particles at all (e.g.
quantum mechanics where everything is a wave function).  These two applications
are beyond the scope of this post.

The Hamiltonian also has some nice properties that aren't obvious at first
glance.  There are three properties that we'll care about:

**Reversability**: An interesting result is that for a particle given its
initial point in phase space :math:`(q_0, p_0)` at a point in time, its motion
is completely determined for all time.  That is, we can use Hamiltonian's
equations to find its instantaneous rate of change (:math:`(q', p')`), which we
can use to find its nearby position after a delta of time, and then repeat this
process to find its trajectory.  This hints at the application we're going to
use it for: using a numerical method to find its trajectory (next subsection).
Equally important though is the fact that we can reverse this process to find
where it came from.  If you have a path from :math:`(q(t), p(t))` to 
:math:`(q(t+s), p(t+s)` then you can find the reverse path by applying the negative
time derivative (:math:`(-q', -p')`) because the path is unique.
We'll use this property when constructing the Markov chain transitions for HMC.

**Conservation of the Hamiltonian**: Another important property is that it
keeps the Hamiltonian conserved.  We can see this by taking the time derivative
of the Hamiltonian (in 1D to keep things simple):

.. math::

   \frac{dH}{dt} &= \frac{dq}{dt}\frac{\partial H}{\partial q} + \frac{dp}{dt}\frac{\partial H}{\partial p} \\
    &= \frac{dq}{dt}\frac{dp}{dt} - \frac{dp}{dt}\frac{dq}{dt} && \text{Hamilton's equations} \\
    &= 0 \\
    \tag{28}

This important property lets us *almost* get to a 100% acceptance rate for HMC.
We'll see later that this ideal is not always maintained.

**Volume preservation**: The last important property we'll use it called
Liouville's theorem (from [2]):

    **Liouville's Theorem**: Given a system of :math:`N` coordinates :math:`q_i`,
    the :math:`2N` dimentional "volume" enclosed by a given :math:`(2N-1)`
    dimensional "surface" in phase space is conserved (that is, independent of
    time) as the surface moves through phase space.
   
I'll refer to [2] if you want to see the proof.  This is an important result
that we'll use so that we can avoid accounting for the change in volume 
(via Jacobians) in our HMC algorithm since the multi-dimensional "volume" is
preserved.  More on this later.

Discretizing Hamiltonian's Equations
------------------------------------

The simple examples we saw in the last subsections worked out nicely where
we had a closed form solution to the equations of motion.  As you can imagine,
in most cases we won't have such a nice closed form analytic solution.  In these
cases, we turn to approximate methods to compute our desired result.

One way to approach this is to iteratively simulate Hamilton's equation by
discretizing time using some small :math:`\epsilon`.  Starting at time 0,
we can iteratively compute the trajectory in phase space :math:`(q, p)`
through time using Hamilton's equations.  We'll look at 2.5 methods to
accomplish this.

**Euler's Method**: `Euler's method <https://en.wikipedia.org/wiki/Euler_method>`__ 
is a technique to solve first order differential equations.  Notice that 
Hamilton's equations produce 2N first order differential equations (as opposed
to the Lagrangian, which produces second order differential equations).
It's essentially just applying a first order Taylor series approximation
at each iteration about the current point.

More precisely, for a given step size :math:`\epsilon`, we can approximate the
curve :math:`y(t)` given an initial point :math:`y_0` and a first order
differential equation using the formula:

.. math::

    y(t+\epsilon) = y(t) + \epsilon y'(t, y(t))  \tag{29}
    
where :math:`y(t_0)=y_0`.  This is simply taking small step sizes along the
gradient of our curve where the gradient is computed from our differential
equation using the :math:`t` and the previous values of `y`.

Translating this to phase space and using Hamilton's equations, we have:

.. math::

   p(t+\epsilon) = p(t) + \epsilon \frac{dp}{dt}(t) = p(t) - \epsilon \frac{\partial H}{\partial q}(q(t)) && \text{by Hamilton's Equation} \\
   q(t+\epsilon) = q(t) + \epsilon \frac{dq}{dt}(t) = q(t) + \epsilon \frac{\partial H}{\partial p}(p(t)) && \text{by Hamilton's Equation} \\
   \tag{30}

Notice that the equations are dependent on each other, to calculate
:math:`p(t+\epsilon)`, we need both of :math:`(q, p)` and vice versa.

The main problem with Euler's method is that it quickly diverges from the 
actual curve because of the accumulation of errors.  The error propagates
because we assume we start from the somewhere on the curve whereas we're always
some delta away from the curve after the first iteration.  Figure 4 shows
how the method quickly spirals out of control towards infinity even with a
small epsilon with our simple harmonic oscillator from Examples 1-3.

.. figure:: /images/hmc_leapfrog.png
  :width: 100%
  :alt: Leapfrog method to approximate Hamiltonian dynamics
  :align: center

  **Figure 4: Methods to approximate Hamiltonian dynamics: Euler's method, modified Euler's method, and Leapfrog
  using the harmonic oscillator from Examples 1-3.**

**Modified Euler's Method**: A simple modification to Euler's method is to
update :math:`p` and :math:`q` separately.  First update :math:`p`,
then use that result to update :math:`q` and repeat (the other way around also
works).  More precisely, we get this approximation in phase space:

.. math::

   p(t+\epsilon) = p(t) + \epsilon \frac{dp}{dt}(t) = p(t) - \epsilon \frac{\partial H}{\partial q}(q(t)) \tag{31}\\
   q(t+\epsilon) = q(t) + \epsilon \frac{dq}{dt}(t) = q(t) + \epsilon \frac{\partial H}{\partial p}(p(t+\epsilon)) \tag{32}

The results can be seen in Figure 4: it more closely tracks the underlying
curve without tendencies to diverge. 

This is because the pair of equations
preserves volume just like the result from Liouville's theorem above.  Let's
show how that is the case in two dimensions but this result holds for multiple
dimensions. (In fact, the argument in the following sketch 
can be used to prove Liouville's theorem albeit with more complexity.)

First note that Equation 31 can be viewed as a transformation mapping
:math:`(p(t), q(t))` to :math:`(p(t+\epsilon), q(t))` (same for Equation 32).
Denote this mapping as :math:`\bf f` and let's see how the differentials of the
above change (I'll change all the parameters to superscripts to make the
notation a bit nicer).  First, we can see the transformation for Equation 31 as:

.. math::

    \begin{bmatrix}
    p^{t+\epsilon} \\
    q^t \\
    \end{bmatrix} = {\bf f}\big(
    \begin{bmatrix}
    p^t \\
    q^t \\
    \end{bmatrix}\big) \tag{32}

Next, let's calculate the Jacobian of :math:`\bf f`:

.. math::

    {\bf J_f} &= \begin{bmatrix}
    \frac{\partial \bf f}{\partial p^t} & \frac{\partial \bf f}{\partial q^t}
    \end{bmatrix} \\
    &= \begin{bmatrix}
    \frac{\partial [p^t - \epsilon \frac{\partial H}{\partial q^t}(q^t)]}{\partial p^t} &
    \frac{\partial [p^t - \epsilon \frac{\partial H}{\partial q^t}(q^t)]}{\partial q^t} \\
    \frac{\partial q^t}{\partial p^t} &
    \frac{\partial q^t}{\partial q^t}
    \end{bmatrix} \\
    &= \begin{bmatrix}
    1 &
    -\frac{\partial [\epsilon \frac{\partial H}{\partial q^t}(q^t)]}{\partial q^t} \\
    0 & 1
    \end{bmatrix} \\ \tag{33}

We can clearly see the determinant of the Jacobian is 1.
Next let's see how the infinitesimal volume (or area in this case) changes 
using the `substitution rule <https://en.wikipedia.org/wiki/Integration_by_substitution#Substitution_for_multiple_variables>`__
(this is usually not shown since the determinant of the Jacobian already implies this):

.. math::

    dp^{t+\epsilon} dq^t = |det({\bf J_f})| dp^t dq^t = dp^t dq^t \tag{34}

So we see that the volume is preserved when we take a single step (Equation 31).
We can use the same logic when applying Equation 32 and every subsequent
application of those equations using modified Euler's method.

Figure 5 shows this visually by drawing a small region near the starting points
and then running Euler's method and modified Euler's method.  For the vanilla
Euler's method, you can see the region growing larger with each iteration. This
has the tendency to cause points to spiral out to infinity (since the area of this region
grows, so do the points that define it).  Modified Euler's doesn't have this problem.

.. figure:: /images/hmc_vol_preserve.png
  :width: 100%
  :alt: Visualization of volume presenvation of modified Euler's method
  :align: center

  **Figure 5: Contrasting volume preservation nature of the modified Euler's method vs. Euler's method.**

It's not clear to me that volume preservation in general guarantees that it
won't spiral to infinite, nor that non-volume preservation necessarily
guarantees it will spiral to infinite but it does sure seem to help empirically.
The guarantees (if any) are likely related to the `symplectic nature <https://en.wikipedia.org/wiki/Symplectic_integrator>`__
but I didn't really look into it much further than that.

**Leapfrog Method**: The final method uses the same idea but with an extra *leapfrog* step:

.. math::

   p(t+\epsilon/2) = p(t) - \epsilon/2 \frac{\partial H}{\partial q}(q(t)) \tag{35}\\
   q(t+\epsilon) = q(t) + \epsilon \frac{\partial H}{\partial p}(p(t+\epsilon/2)) \tag{36} \\
   p(t+\epsilon) = p(t+\epsilon/2) - \epsilon/2 \frac{\partial H}{\partial q}(q(t+\epsilon)) \tag{37}

where we iteratively apply these equations sequentially similar to modified Euler's method.
The idea is that instead of taking a "full step" for :math:`p`, we take a "half step". 
This half step is used to update :math:`q`, which is then used to update
:math:`p` using another "half step".  The last subplot in Figure 4 shows Leapfrog, which
empirically performs much better than the other methods.

Using the same logic as above, each transform individually is volume
preserving, ensuring similar "nice" behaviour as modified Euler's method.
Notice we're doing slightly more "work" in that we're evaluating Hamilton's
equations an additional time but the trade-off is good in this case.

Another nice property of both modified Euler's and Leapfrog is that it is also
reversible.  Simply negate :math:`p`, and run the algorithm, then negate
:math:`p` to get back where you started.  Since we're only updating either
:math:`p` or :math:`q`, it allows us to essentially run the algorithm in
reverse.  As we might expect in MCMC (see background section), this
reversibility condition is important to guarantee a stationary distribution.


Hamiltonian Monte Carlo
=======================

Finally we get to the good stuff: Hamiltonian Monte Carlo (HMC)!  
The main idea behind HMC is that we're going to use Hamiltonian dynamics to
simulate moving around our target distribution's density.  The analogy
used in [1] is imagine a puck moving along a frictionless 2D surface [2]_.  It
slides up and down hills, losing or gaining velocity (i.e. kinetic energy)
based on the gradient of the hill (i.e. potential energy).  Sound familiar?
This analogy with a physical system is precisely the reason why Hamiltonian
dynamics is such a good fit.

The mapping from the physical situation to our MCMC procedure will be such
that the variables in our target distribution will correspond to the position
(:math:`q`), the potential energy will be the negative log probability density
of our target distribution, and the momentum variables (:math:`p`) will be
artificially introduced to allow us to sample properly.  So without further
adieu, let's get into the details!

From Thermodynamics to HMC
--------------------------

The base physical system we're going to base this on is from thermodynamics
(which is only slightly more complex than the mechanical systems we're been
looking at).  A commonly studied situation in thermodynamics is the one of
a closed system of fixed volume and number of particles (e.g. gas molecules in
a box) that is "submerged" in a heat bath at thermal equilibrium.
The basic idea is the heat bath is much, much larger than our internal system so
it can keep it the system at a constant temperature.  
Note that even though internal system is at a constant temperature, its energy
will fluctuate because of the mechanical contact with the heat bath, so energy
is not conserved (i.e., constant). The overall system including the heat bath
*and* internal system is conserved though.  The statistical ensemble of this type of
system , also known as the `canonical ensemble <https://en.wikipedia.org/wiki/Canonical_ensemble>`__.

One of the fundamental concepts in this study is the idea of a 
`microstate <https://en.wikipedia.org/wiki/Microstate_(statistical_mechanics)>`__, 
which defines (for classical systems) a single point in phase space.  That is,
the position (:math:`q`) and momentum variables (:math:`p`) for all particles
defines the microstate of the entire system.
We're typically are that interested in the actual movement of particles
(although will be for MCMC), instead we will usually want to measure other
macro thermodynamic quantities such as average energy or pressure of the internal system.

An important quantity we need to compute is the probability of the entire
system being in a microstate i.e., a given configuration of :math:`p`'s and
:math:`q`'s.  Without going into the entire derivation, which would take us on
a larger tangent into thermodynamics, I'll just give the result, which is known
as the Boltzman distribution:

.. math::

   p_i    &= \frac{1}{Z} e^{\frac{E_i}{kT}} && \text{general form}\\
   P(q, p) &= \frac{1}{Z} e^{\frac{H(q, p)}{kT}} && \text{Hamiltonian form} \\
          \tag{38}

where :math:`p_i`  is the probability of being in state :math:`i`, :math:`P(q, p)`
is the same probability but explicitly labeling the state with its phase state coordinates
:math:`(q, p)`, :math:`E_i` is the energy state of state :math:`i`, :math:`k` is the
Boltzmann constant, and :math:`T` is the temperature.  As we know from the previous
section, the total energy of a system is (in this case) equal to the Hamiltonian so
we can easily re-write :math:`E_i` as :math:`H(q, p)` to get the second form.  

It turns out that it doesn't matter how many particles you have in your
internal system, it could be a googleplex or a single particle.  As long as you
have the heat bath and some assumptions about the transfer of heat between the
two systems, the Boltzmann distribution holds for the system.  The most intuitive
way to think about it is (as an ML person) as a "softmax" over all the microstates,
where the energy of the microstate is the "logit" value.  Importantly, it is
*not* just an exponential distribution.

In the single particle case, the particle is going to be moving around in your
closed system but randomly interacting with the heat bath, which basically
translates to changing its velocity (or momentum).  This is an important idea
that we're going to use momentarily.


.. admonition:: Example 4: Example of canonical ensemble for a classical system with a particle in a potential well.

    .. figure:: /images/hmc_canonical_ensemble.png
      :width: 50%
      :alt: Example of canonical ensemble for a classical system with a particle in a potential well.
      :align: center
    
      **Figure 6: Example of canonical ensemble for a classical system with a
      particle in a potential well. (source: Wikipedia)**
   
    Figure 6 shows a simple 1 dimensional classical (i.e., non-quantum) system
    where a particle is trapped inside a potential well.  The system is
    submerged in a heat bath (not-shown) to keep it in thermal equilibrium.
    The top diagram shows the momentum vs. position, in other words
    it plots the phase space coordinates :math:`(p, x)`.  The bottom left plot shows
    the energy of the system vs. position with the red line indicating the potential
    energy at each :math:`x` value.  The bottom right plot shows the distribution
    of states across energy levels.
    
    A few things to point out:
     
    * The particle moves along a single axis denoted by the position :math:`x`.
      So it essentially just moves left and right.
    * The velocity (or momentum) changes in two ways: (a) As it moves left and
      right, it gains or loses potential energy. This translates into kinetic
      energy affecting the velocity (and momentum).  As it approaches an
      potential "uphill" its movement along the 1D axis slows in that
      direction, similarly when on a potential "downhill" its movement speeds
      up along the 1D axis in that direction.
      (b) The heat bath will be constantly exchanging energy with the system,
      which translates to changing the momentum of the particle.  This happens
      randomly as a function of the equilibrium temperature.
    * The top phase space plot clearly shows the particle spending most of its
      paths (blue) in the dips in the potential function with varying momentum values.
      This is as expected because the particle will get "pulled" into the dips
      while the momentum could vary by the interaction with the heat bath.
    * The bottom left plot shows something similar where the particle is more concentrated
      in the dips of the potential function.  Additionally, most of the time
      the system energy is close to the green dotted line, which represents the average
      energy of the particle system.
    * The bottom right plot shows the distribution of states by energy.  Note that the
      energy states are not a simple exponential distribution as you may think
      from Equation 38.  The distribution in Equation 38 is a function of the
      microstates :math:`(q, p)`, *not* the system energy.  
      This is hidden in the normalization constant :math:`Z`, which sums over all
      microstates to normalize the probabilities to 1.  As a result, the distribution
      over energy states can be quite complex as shown.
   
As we can see from Equation 38 and Example 4, we have related the Hamiltonian
to a probability distribution.  We now (finally!) have everything we need to
setup the HMC method.

This whole digression into thermodynamics is not for naught!  We are in fact
going to use the canonical ensemble to model in order to sample our target
distribution.  Here's the setup for target density :math:`f({\bf x})` with
:math:`D` variable in its support:

* **Position variables** (:math:`q`): The :math:`D` variables of our target
  distribution (the one we want to sample from) will correspond to our position
  variables :math:`\bf q`.  Instead of our canonical distribution existing in
  (usually) 3 dimensions, we'll be using :math:`D` position dimensions.
* **Momentum variables** (:math:`p`): :math:`D` corresponding momentum
  variables will be introduced artificially in order for the Hamiltonian
  dynamics to operate.  They will allow us to simulate the particle moving
  around as well as it randomly changing position when it interacts with the
  heat bath.
* **Potential energy** (:math:`U(q)`): The potential energy will be the
  negative logarithm of our target density (up to a normalizing constant):

  .. math::

        U({\bf q}) = -log[f({\bf q})] \tag{39}
* **Kinetic energy** (:math:`K(p)`): There can be many choices in how to define
  the kinetic energy, but the current practice is to assume that it is independent
  of :math:`q`, and its quadratic in each of the dimensions.  This naturally
  translates to a zero-mean multivariate Gaussian (see below), which is usually
  specified to be independent with variance :math:`m_i`.  This produces the
  kinetic energy:

  .. math::

        K({\bf p}) = \sum_{i=1}^D \frac{p_i^2}{2m_i} \tag{40}
* **Hamiltonian** (:math:`H({\bf q, p})`): Equation 39 and 40 imply that this Hamiltonian:

  .. math::

        H({\bf q, p}) = -log[f({\bf q})] + \sum_{i=1}^D \frac{p_i^2}{2m_i} \tag{41}
* **Canonical distribution** (:math:`P({\bf q, p})`): The canonical ensemble
  yields the Boltzmann equation from Equation 38 where we will set :math:`kT=1`
  and plug in our Hamiltonian from Equation 40:

  .. math::

        P({\bf q, p}) &= \frac{1}{Z}\exp(\frac{H({\bf q, p})}{kT}) && \text{set } kT=1\\
                      &= \frac{1}{Z}\exp(-log[f({\bf q})] + \sum_{i=1}^D \frac{p_i^2}{2m_i}) \\
                      &= \frac{1}{Z_1}\exp(-log[f({\bf q})])\cdot\frac{1}{Z_2}\exp(\sum_{i=1}^D \frac{p_i^2}{2m_i}) \\
                      &= P(q)P(p)
        \tag{42}

where :math:`Z_1, Z_2` are normalizing constants, and :math:`P(q), P(p)` are
independent distributions involving only those variables.  Taking a closer
look at those two distributions, we have:

.. math::


    P({\bf q}) = \frac{1}{Z_1}\exp(-log[f({\bf q})]) = \frac{1}{Z_1} f({\bf q}) \propto f({\bf q}) \\
    P({\bf p}) = \cdot\frac{1}{Z_2}\exp(\sum_{i=1}^D \frac{p_i^2}{2m_i}) \\
    \tag{43}

So our canonical distribution is made up of two independent parts: our target distribution
and a zero mean Gaussian!  So how does this help us?  Recall that the canonical distribution
models the distribution of microstates (:math:`\bf q,p`), so if we can *exactly* simulate the
dynamics of the system (via the Hamilton's equations + random interactions with
the heat bath), we would essentially be simulating exactly :math:`P({\bf q,p})`, which
leads us directly to simulating :math:`P({\bf q})`!

.. admonition Why do we need to model the random interactions with the heat bath?

   There are two ways to think about this problem.  The first is that if want
   to use the Boltzmann distribution, the assumptions only hold either for a
   system enclosed in a heat bath *or* if it's a closed system with a very large
   number of particles.  Obviously our single particle model only fits into the
   former.  If we exclude the heat bath then there is an alternate distribution
   specified by the `microcanonical ensemble <https://en.wikipedia.org/wiki/Microcanonical_ensemble>`__.

   Another way to understand it is from the perspective using MCMC to sample
   our target distribution.  If we didn't model the random interactions, the
   total energy of the system would be fixed (:math:`H(q,p)` is constant).
   Therefore, there is a possibility that we would never be able to reach
   certain states with a greater energy level, resulting in the procedure not
   able to sample parts of the target distribution's support.  Obviously, this
   would not lead to a correct sampling procedure.

In this hypothetical scenario, we would just need to simulate this system, record
our :math:`q` values, and out would pop samples of our target distribution.
Unfortunately, this is not possible.  The main reason is that we cannot *exactly*
simulate this system because, in general, Hamilton's equations do not yield a
closed form solution.  So we'll have to discretize Hamiltoninan dynamics and add 
in an Metrpolis-Hastings update step to make sure we're faithfully simulating our
target distribution.  The next subsection describes the HMC algorithm in more detail.

HMC Algorithm
-------------

The core part of the HMC algorithm follows essentially the same structure as
the Metropolis-Hastings algorithm: propose a new sample, accept with some
probability.  The difference is that Hamiltonian dynamics are used to find a
new proposal sample, and the acceptance criteria is slightly modified.
Here's a run-down of the major steps:

1. Draw a new value of :math:`p` from our zero mean Gaussian.  This simulates
   a random interaction with the heat bath.
2. Starting in state :math:`(q,p)`, run Hamiltonian dynamics for :math:`L` steps
   with stepsize :math:`\epsilon` using the Leapfrog method presented in
   Section 2.6.  :math:`L` and :math:`\epsilon` are hyperparameters of the
   algorithm.  This simulates the particle moving without interactions with the heat bath.
3. After running :math:`L` steps, negate the momentum variables, giving a proposed
   state of :math:`(q*, p*)`.  This makes the proposed state symmetric i.e.  if
   we run :math:`L` steps again, we get back to the same original state.  The
   negation is necessary for our MCMC proof below but the :math:`p*` value is
   never actually used.
4. The proposed state :math:`(q*, p*)` is accepted as the next state using a
   Metropolis-Hastings-like update with probability:

   .. math::

       A((q*, p*)) &= \min[1, \frac{\exp(-H(q*, p*))}{H(q,p))}] \\
                   &= \min[1, \exp(-U(q*) + U(q) -K(p*)+K(p))] \\
                   \tag{44}
  
   If the next state is not accepted (i.e. rejected), then the current state
   becomes the next state.  This MH step is needed to offset the approximation
   of our discretized Hamiltonian.  If we could exactly simulate Hamiltonian
   dynamics this acceptance probability would be exactly :math:`1` because the
   Hamiltonian is conserved (i.e. constant).

It's all relatively straight forward (assuming you have the requisite
background knowledge above).  It's generally converges faster than
a random walk-based MH algorithm, but it does have some key assumptions.
First, we can only sample from continuous distributions on
:math:`\mathcal{R}^D` because otherwise our Hamiltonian dynamics could not
operate.  Second, similarly to MH, we need to be able to evaluate the density
up to a normalizing constant.  Finally, we must be able to compute the partial
derivative of the log density in order to compute Hamilton's equations.  Thus,
these derivatives must exist everywhere the density is non-zero.
There are a couple of other details you can look up in [1] if you are interested.

What's nice is that all that math reduces down to quite a simple algorithm.
Listing 1 shows pseudo-code for one iteration of the algorithm, which is pretty
straightforward to implement (see the next section where I implement a toy
version of HMC).

**Listing 1: Hamiltonian Monte Carlo Python-like Pseudocode**

.. code-block:: python
   :number-lines:

   def hmc_iteration(U, grad_U, epsilon, L, current_Q, std_dev):
   '''
        U: function returns the potential energy given a state q
        grad_u: function returns gradient of U given q
        epsilon: step size
        L: number of leapfrog steps
        current_Q: current generalized state trajectory starts from
        std_dev: vector of standard deviations for Gaussian (hyperparameter)
   '''
       q = current_q
       p = sample_normal(length(q), 0, std_dev) # sample zero-mean Gaussian
       current_p = p

       # Leapfrog: half step for momentum
       p = p - epsilon * grad_U(q) / 2

       for i in range(0, L):
           # Leapfrog: full step for position
           q = q + epsilon * p

           # Leapfrog: combine 2 half-steps for momentum across iterations
           if (i != L-1):
               p = p - epsilon * grad_U(q)

       # Leapfrog: final half step for momentum
       p = p - epsilon * grad_U(q)

       # Negate trajectory to make proposal symmetric (a no-op)
       p = -p

       # Compute potential and kinetic energies
       current_U = U(current_q)
       current_K = sum(current_p^2) / 2
       proposed_U = U(q)
       proposed_K = sum(p^2) / 2

       # Accept with probability specified using Equation 44:
       if rand(0, 1) < exp(current_U - proposed_U + current_K - proposed_K):
           return q
       else:
           return current_q

Listing 1 is a straight forward implementation of leapfrog combined with a
simple acceptance step. One big of optimization is on line 23 to combine 
the two half momentum steps from Equation 35 and 37.  In the leapfrog algorithm,
every half momentum step except the first and last can be combined into a full
step.  A bit of the magic is hidden behind the potential and gradient of the
potential function but those depend fully on your target distribution so it
can't be helped.

It's not obvious that the above algorithm would be correct, particularly the
acceptance step, which we simply stated without much reasoning.  We'll examine
its correctness the next subsection.

HMC Algorithm Correctness
-------------------------

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

.. [1] The usual symbols they use for the Lagrangian are :math:`L = T - U` representing the kinetic and potential energy respectively.  However, :math:`T` makes no sense to me, so since we're not really talking about physics here, I'll just use :math:`K` to make it clear for the rest of us.
.. [2] This physical analogy is not exactly accurate because gravity, which affects the velocity of the puck, doesn't quite match our target density.  Instead, a better analogy would be a particle moving around in a vector field (e.g. an electron moving around in an electric field defined by our target density).  Although more accurate, it's less intuitive than a puck sliding along a surface so I get why the other analogy is better.
