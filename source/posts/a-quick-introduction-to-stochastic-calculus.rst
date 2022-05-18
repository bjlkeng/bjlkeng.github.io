.. title: A Quick Introduction to Stochastic Calculus
.. slug: a-quick-introduction-to-stochastic-calculus
.. date: 2022-04-29 21:05:55 UTC-04:00
.. tags: stochastic calculus, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

Write your post here.


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

Stochastic Processes
====================

Probability Spaces & Random Variables
-------------------------------------

(Skip this part if you're already familiar with the measure-theoretic probability definition.)

First, let's examine the definition of a **probability space** :math:`(\Omega, {\mathcal {F}}, P)`.
This is basically the same setup you learn in a basic probability class, except
with fancier math.

:math:`\Omega` is the **sample space**, which defines the set
of all possible outcomes or results of that experiment.  In finite sample
spaces, any subset of the samples space is called an **event**.  Another way to
think about events is any thing you would want to measure the probability on,
e.g. individual elements of :math:`\Omega`,  unions of elements, or even the
empty set.

However, this type of reasoning breaks down when we have certain types of
infinite samples spaces (e.g. real line).  For this, we need to define an events more precisely 
with an **event space** :math:`\mathcal{F} \subseteq 2^{\Omega}` using a construction called a :math:`\sigma`-algebra
("sigma algebra").  This sounds complicated but it basically is guaranteeing
that the subsets of :math:`\Omega` that we use for events are 
`measurable <https://en.wikipedia.org/wiki/Measure_(mathematics)>`__
(this makes the notion of "size" or "volume" precise, ensuring that
the size of the union of disjoint sets equals their sum -- exactly what 
we need for probabilities).

Which brings us to our the last part of probability spaces: a **probability
measure** :math:`P` on :math:`\mathcal{F}` is a function that maps events to
the unit interval :math:`[0, 1]` and returns :math:`0` for the empty set and
:math:`1` for the entire space (it must also satisfy countable additivity).
Again, for finite sample spaces, it's not too hard to imagine this function and
how to define it but, as you can imagine, for continuous sample spaces, it gets
more complicated.  All this is essentially to define a rigorous construction
that matches our intuition of basic probability with samples spaces, events,
and probabilities.

Finally, a **random variable** :math:`X` is a `measurable function <https://en.wikipedia.org/wiki/Measurable_function>`__
:math:`X:\Omega \rightarrow \mathbb{R}` (with :math:`\Omega` from a given
probability space) [1]_.  When accompanied by the corresponding measure for the probability
space :math:`P`, one can calculate the probability of :math:`x \in \mathbb{R}` (and its corresponding
its distribution) as: 

.. math::

    P(X = x) = P(\{\omega \in \Omega | X(\omega) = x \}) \tag{1}

Basically, a random variable allows us to map to real numbers from our original
sample space.  So if our sample space has no concept of numbers (e.g. heads or
tails), this allows us to assign real numbers to those events to calculate
things like expected values and variance.  To find the associated probability,
we map backwards from our real number (:math:`x`) to a set of values in the
sample space (:math:`\omega`) using :math:`X` forming an event, and then 
from this event we can calculate the probability using the original definition
above.


.. admonition:: Example 1: Sample Spaces, Events, Probability Measures, and Random Variables

   (From `Wikipedia <https://en.wikipedia.org/wiki/Event_(probability_theory)#A_simple_example>`__)

   Assume we have a standard 52 card playing deck without any jokers,
   and our experiment is that we draw a card randomly from this set.
   The sample space :math:`\Omega` is a set consisting of the 52 cards.
   An event :math:`A \subseteq \mathcal{F}` is any subset of :math:`\Omega`.
   So that would include the empty set, any single element, or even the entire
   sample space.  Some examples of events:

   * "Cards that are red and black at the same time" (0 elements)
   * "The 5 of Hearts" (1 element)
   * "A King" (4 elements)
   * "A Face card" (12 elements)
   * "A card" (52 elements)

   In the case where each card is equally likely to be drawn, we 
   can define a probability measure for event :math:`A` as:
   
   .. math::

        P(A) = \frac{|A|}{|\Omega|} = \frac{|A|}{52} \tag{2}

   We can additionally define a random variable:
   
   .. math::

        X(\omega \in \Omega) = 
        \begin{cases}
            1 &\text{if } x \text{ is red}\\
            0 &\text{otherwise}
        \end{cases}
        \tag{3}

   We can calculate probabilities using Equation 1, for example :math:`X = 1`:

   .. math::
        
        P(X = 1) &= P(\{\omega \in \Omega | X(\omega) = 1 \}) \\
        &= P(\{\text{all red cards}\})  \\
        &= \frac{|\{\text{all red cards}\}|}{52} \\
        &= \frac{1}{2}  \\
        \tag{4}

.. admonition:: The Two Stages of Learning Probability Theory 

    *(Inspired by the notes from Chapter 1 in [1])*

    Probability theory is generally learned in two stages.  The first stage
    describes discrete random variables that have a probability mass function,
    and continuous random variables that have a density.  We learn to compute
    basic quantities from these variables such as expectations, variances, 
    and conditionals.  We learn about standard distributions and their properties
    and how to manipulate them such as 
    `transforming continuous random variables <https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function>`__.
    This gets us through most of the standard applications of probability
    from basic statistical tests to likelihood functions.

    The second stage of probability theory dives deep into the rigorous
    measure-theoretic definition.  In this definition, one views a 
    random variable as a function from a sample space :math:`\Omega`
    to a set of real numbers :math:`\mathbb{R}`.  Certain subsets
    of :math:`\Omega` are called events, and the collection of all possible
    events form a :math:`\sigma`-algebra :math:`\mathcal {F}`.  Each
    set :math:`A` in :math:`\mathcal {F}` has probability :math:`P(A)`, 
    defined by the probability measure :math:`P`.
    This definition handles both discrete and continuous variables in a elegant
    way.  It also (as you would expect) introduces a lot of details underlying
    the results that we learn in the first stage.  For example, a random
    variable is not the same thing as a distribution (random variables can have
    multiple probability distributions depending on the associated probability
    measure).  Another quirk that we often don't think about is that not all
    distributions have a density function (although most of the distributions
    we study will have a density).  Like many things in applied mathematics, 
    understanding of the rigorous definition is often not needed because
    most of the uses do not hit the corner cases where it matters (until it
    doesn't).  It's also a whole lot of work to dig into so most folks
    like me are happy to understand it only "to a satisfactory degree".


Definition
----------

Here's the formal definition of a 
`stochastic process <https://en.wikipedia.org/wiki/Stochastic_process#Stochastic_process>`__ from [2]:

    Suppose that :math:`(\Omega,\mathcal{F},P)` is a probability space, and that :math:`T \subset \mathbb{R}`
    is of infinite cardinality. Suppose further that for each :math:`t \in T`, 
    there is a random variable :math:`X_t: \Omega \rightarrow \mathbb{R}` 
    defined on :math:`(\Omega,\mathcal{F},P)`. The function :math:`X: T \times \Omega \rightarrow \mathbb{R}` 
    defined by :math:`X(t, \omega) = X_t(\omega)` is called a stochastic process with
    indexing set :math:`T`, and is written :math:`X = \{X_t, t \in T\}`.


That's a mouthful!  Let's break this down and interpret the definition more intuitively.
We've already seen probability spaces and random variables in the previous
subsection.  The first layer of a stochastic process is that we have a bunch of
random variables that are indexed by some set :math:`T`.  Usually :math:`T` is
some total ordered sequence such as a subset of the real line (e.g. :math:`(0,
\infty)`) or natural numbers (e.g. :math:`0, 1, 2, 3 \ldots`), which intuitively
correspond to continuous and discrete time.

Next, we turn to the probability space on which each random variable is defined on
:math:`(\Omega,\mathcal{F},P)`.  The key thing to note is that the elements of 
the sample space :math:`\omega \in \Omega` are infinite sets that correspond to
experiments performed at each index in :math:`T`.  Note: by definition it's infinite
because otherwise it would just be a random vector.  For example, flipping a 
coin at every (discrete) time from :math:`0` to :math:`\infty`, would define a
specific infinite sequence of heads and tails :math:`\omega = \{H, T, H, H, H, T, \ldots\}`.
So each random variable :math:`X_t` can depend on the entire sequence of the
outcome of this infinite "experiment".  That is, :math:`X_t` is a mapping
from outcomes of our infinite experiment to the real numbers: 
:math:`X_t: \Omega \rightarrow \mathbb{R}`.  (Recall to get the probability for a
value of :math:`X_t` we would need to map the real number back to the sample space,
then use the probability measure :math:`P` shown in Equation 1.)
It's important to note that in this general definition we have no explicit
concept of time, so we can depend on the "future".  To include our usual
concept of time, we need an additional concept (see adapted below).

Finally, instead of viewing the stochastic process as a collection of random variables
indexed by time, we could look at it as a function of both time and the sample space
i.e., :math:`X(t, \omega) = X_t(\omega)`.  For a given outcome of an experiment
:math:`\omega`, the deterministic function generated as :math:`X(t, \omega)` is
called the **sample function**.  However, mostly we like to think of it
as having a random variable at each time step indicated by this notation: 
:math:`X = \{X_t, t \in T\}`.  We sometimes us the notation :math:`X(t)` to refer
to the random variable at time :math:`t` or the stochastic process itself.

Stochastic processes can be classified by the nature of the values the random variables
take and/or the nature of the index set:

* **Discrete and Continuous Value Processes**: :math:`X(t)` is discrete if at all "times" :math:`X(t)` takes on values in a 
  `countable set <https://en.wikipedia.org/wiki/Countable_set>`__ (i.e., can be mapped to a subset of the natural numbers);
  otherwise :math:`X(t)` is continuous.
* **Discrete and Continuous Time Processes**: :math:`X(t)` is discrete time process if the index set is 
  countable (i.e., can be mapped to a subset of the natural numbers).

Generally continuous time processes are harder to analyze and will be the focus
of later sections.  The next two discrete time examples give some intuition about
how to match the formal definition to concrete stochastic processes.

.. admonition:: Example 2: Bernoulli Processes

    One of the simplest stochastic processes is a 
    `Bernoulli Process <https://en.wikipedia.org/wiki/Bernoulli_process>`__, which
    is a discrete value, discrete time process.  The main idea is that a
    Bernoulli process is a sequence of independent and identically distributed
    Bernoulli trials (think coin flips) at each time step.
  
    More formally, our sample space :math:`\Omega = \{ (a_n)_1^{\infty} : a_n
    \in {H, T} \}`, that is, the set of all infinite sequences of "heads" and "tails".
    It turns out the event space and the probability measure are surprisingly
    complex to define so I've put those details in Appendix A.

    We can define the random variable given an outcome of infinite tosses
    :math:`\omega`:

    .. math::

        X_t(\omega) =  \begin{cases}
            1 &\text{if } \omega_t = H\\
            0 &\text{otherwise}
        \end{cases} \tag{5}

    for :math:`\omega = \omega_1 \omega_2 \omega_3 \ldots`, where each :math:`\omega_i`
    is the outcome of the :math:`i^{th}` toss.
    For all values of :math:`t`, the probability :math:`P(X_t = 1) = p`, for
    some constant :math:`p \in [0, 1]`.

.. admonition:: Example 3: One Dimensional Random Walk

   A simple one dimensional `random walk <https://en.wikipedia.org/wiki/Random_walk>`__
   is a discrete value, discrete time stochastic process.  An easy way to 
   think of it is: starting at 0, at each time step, flip a fair coin and move
   right (+1) if heads, otherwise move left (-1).

   This can be defined using the same probability space as the Bernoulli process from Example 2
   with :math:`p=0.5` but with a different definition of the random variable at each time step:

   .. math::

        X_t(\omega) =  \sum_{i=1}^t x_i \text{ for } x_i
        \begin{cases}
            1 &\text{if } \omega_i = H\\
            -1 &\text{otherwise}
        \end{cases} \tag{6}

Adapted Processes
-----------------

Notice that in the previous section, our definition of stochastic process
included a random variable :math:`X_t: \Omega \rightarrow \mathbb{R}`
where each :math`\omega \in \Omega` is an infinite set representing a
given outcome for the infinitely long experiment.  This implicitly means
that at "time" :math:`t`, we could depend on the "future".  In many
applications, we do want to interpret :math:`t` as time so we wish
to restrict our definition of stochastic processes.

An `adapted stochastic process <https://en.wikipedia.org/wiki/Adapted_process>`__
is one that cannot "see into the future".  Informally, it means that for
any :math:`X_t`, you can determine it's value by *only* seeing the outcome 
of the experiment up to time :math:`t`.  

More formally, we need to introduce the concept of a filtration on our
event space :math:`\mathcal{F}` (i.e., :math:`\sigma`-algebra) and our index
set :math:`T`:

    A **filtration** :math:`\mathbb{F}` is a ordered collection
    of subsets :math:`\mathbb{F} := (\mathcal{F_t})_{t\in T}` where 
    :math:`F_t` is a sub-:math:`\sigma`-algebra of :math:`\mathcal{F}`
    and :math:`\mathcal{F_{t_1}} \subseteq \mathcal{F_{t_2}}` for all
    :math:`t_1 \leq t_2`.

To break this down, we're basically saying that our event space :math:`\mathcal{F}`
can be broken down into logical "sub event spaces" :math:`\mathcal{F_t}` such
that each one is a superset of the next one.  This is precisely what we want
where as we progress through time, we "gain" more information but never lose
any.  We also use this idea of defining a sub-:math:`\sigma`-algebra to
formally define conditional probabilities.

Using the construct of a filtration, we can define a stochastic process
:math:`X_t : T \times \Omega` that is **adapted to the filtration**
:math:`(\mathcal{F_t})_{t\in T}` if the random variable :math:`X_t`
is a :math:`(F_t, \Sigma)` measurable function.  This basically says
that :math:`X_t` can only depend on outcomes before or at time :math:`t`
(with the definition of "outcomes" very loosely defined by the filtration).
As with much of this topic, we require a lot of rigour in order to make
sure we don't have weird corner cases that violate the 
`soundness <https://en.wikipedia.org/wiki/Soundness>`__ of the theory.
The next example gives more intuition on adapted processes.

.. admonition:: Example 2: An Adapted Bernoulli Processes

    TODO: Use this example: https://en.wikipedia.org/wiki/%CE%A3-algebra#Sub_%CF%83-algebras


* Adapted Processes: https://en.wikipedia.org/wiki/Adapted_process
  * Itō integral, which only makes sense if the integrand is an adapted process. 

Weiner Processes
----------------

* Define (Wikipedia, Hull textbook)
* Basic properties
* Continuous everywhere, Differentiable nowhere
* Quadratic variation?
* Surely p=1.0 to return to value, and it's length is infinite
* Example (use something from Hull textbook)

Stochastic Integrals
====================

* Stochastic integral (see lectures notes "A Quick introduction to stochastic calculus")
    * Why we need it? non-differentiable
    * Use the basic Brownian motion integral as an example
* Different types of stochastic calculus' you can come up with depending on definition

Stochastic Differential Equations (SDE)
=======================================

* dX = adt + bdB
* https://en.wikipedia.org/wiki/Stochastic_differential_equation
* Ito Processes https://en.wikipedia.org/wiki/It%C3%B4_calculus#It%C3%B4_processes

Itô's Lemma
===========
* https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma
* Simple derivation
* Examples: 

Applications
============

Stock Prices and the Black-Scholes Equation
-------------------------------------------
* Stock prices
* Black-Scholes Equation

Application: Langevin Equation
------------------------------
* Langevin Equation
  * https://en.wikipedia.org/wiki/Langevin_equation#Trajectories_of_free_Brownian_particles
  * https://en.wikipedia.org/wiki/Langevin_equation#Recovering_Boltzmann_statistics


References
==========
* Wikipedia: `Stochastic Processes <https://en.wikipedia.org/wiki/Stochastic_process#Stochastic_process>`__, `Adapted Stochastic Process <https://en.wikipedia.org/wiki/Adapted_process>`__
* [1] Steven E. Shreve, "Stochastic Calculus for Finance II: Continuous Time Models", Springer, 2004.
* [2] Michael Kozdron, "`Introduction to Stochastic Processes Notes <https://uregina.ca/~kozdron/Teaching/Regina/862Winter06/Handouts/revised_lecture1.pdf>`__", Stats 862, University of Regina, 2006.


Appendix A: Event Space and Probability Measure for a Bernoulli Process
=======================================================================

As mentioned the sample space for the Bernoulli process is all infinite
sequences of heads and tails: :math:`\Omega = \{ (a_n)_1^{\infty} : a_n \in {H, T} \}`.
The first thing to mention about this sample space is that it is
`uncountable <https://en.wikipedia.org/wiki/Uncountable_set>`__,
which basically means it is "larger" than the natural numbers.
Reasoning in infinities is quite unnatural but the two frequent "infinities"
that usually pop up are sets that have the same 
`cardinality <https://en.wikipedia.org/wiki/Cardinality>`__ ("size") as
(a) the natural numbers, and (b) the real numbers.
For our sample space has the same cardinality as the latter.
Cantor's original diagonalization argument 
`diagonalization argument <https://en.wikipedia.org/wiki/Cantor%27s_diagonal_argument>`__
actually used a variation of this sample space (with :math:`\{0, 1\}`'s), and
the proof is relatively intuitive.  
In any case, this complicates things because a lot of our intuition falls apart
when we work with infinites, and especially with infinities the size of the
real numbers.

*(This construction was taken from [1], which is a dense, but informative reference for all the topics in this post.)*

Now we will construct the event space (:math:`\sigma`-algebra) and probability
measure for the Bernoulli process.  We'll do it iteratively.  First, let's define
:math:`P(\emptyset) = 0` and :math:`P(\Sigma) = 1`, and the corresponding (trivial)
event space: 

.. math::

    \mathcal{F}_0 = \{\emptyset, \Sigma\} \tag{A.1}
  
Notice that :math:`\mathcal{F}_0` is a :math:`\sigma`-algebra.  Next, let's
define two sets: 

.. math::

   A_H &= \text{the set of all sequences beginning with } H = \{\omega: \omega_1 = H\} \\
   A_T &= \text{the set of all sequences beginning with } T = \{\omega: \omega_1 = T\} \\
   \tag{A.2}

And set the intuitive definition of the corresponding probability measure:
:math:`P(A_H) = p` and :math:`P(A_T) = 1-p`.  That is, the probability of
seeing an H on the first toss is :math:`p`, otherwise :math:`T`.
Since these two sets are compliments of each other (:math:`A_H = A_T^c`),
this defines another :math:`\sigma`-algebra:

.. math::

    \mathcal{F}_1 = \{\emptyset, \Sigma, A_H, A_T\} \tag{A.3}

We can repeat this process again but for the first two tosses, define sets:

.. math::

   A_{HH} &= \text{the set of all sequences beginning with } HH = \{\omega: \omega_1\omega_2 = HH\} \\
   A_{HT} &= \text{the set of all sequences beginning with } HT = \{\omega: \omega_1\omega_2 = HT\} \\
   A_{TH} &= \text{the set of all sequences beginning with } TH = \{\omega: \omega_1\omega_2 = TH\} \\
   A_{TT} &= \text{the set of all sequences beginning with } TT = \{\omega: \omega_1\omega_2 = TT\} \\
   \tag{A.4}

Similarly, we can extend our probability measure with the definition we would expect:
:math:`P(A_{HH}) = p^2, P(A_{HT}) = p(1-p), P(A_{TH}) = p(1-p), P(A_{TT}) = (1-p)^2`.
Now we have to do a bit more analysis, but if one works out every possible set we can
create either from complimentation or union of any of the above sets, we'll find
that we have 16 in total.  For each one of them, we can compute its probability
measure by using one of the above definitions or by the fact that :math:`P(A) = 1-P(A)`
or :math:`P\big(\bigcup_{n=1}^{N} A_N \big) = \sum_{n=1}^{N} P(A_N)` if the sets
are disjoint.  These 16 sets define our next :math:`\sigma`-algebra:

.. math::

    \mathcal{F}_2 = \left. \begin{cases}
            \emptyset, \Sigma, A_H, A_T, A_{HH}, A_{HT}, A_{TH}, A_{TT}, A_{HH}^c, A_{HT}^c, A_{TH}^c, A_{TT}^c \\
            A_{HH} \bigcup A_{TH}, A_{HH} \bigcup A_{TT}, A_{HT} \bigcup A_{TH}, A_{HT} \bigcup A_{TT}
        \end{cases} \right\} \tag{A.5}

As you can imagine, we can continue this process and define the probability (and associated 
:math:`\sigma`-algebra) for every set in terms of finitely many tosses.  Let's call
this set :math:`\mathcal{F}_\infty`, which contains all of the sets that can be described
by finitely many coin tosses using the procedure above, and then adding in all the
other ones using the compliment or union operator.  This turns out to be precisely
the :math:`\sigma`-algebra: of the Bernoulli process.  And by the construction, 
we also have defined the associated probability measure for each one of the events
in :math:`\mathcal{F}_\infty`.

Now we could leave it there, but let's take a look at the non-intuitive things that go
on when we work with infinities.  This definition implicitly includes sequences
that weren't explicitly defined by us, for example, the sequence of all heads:
:math:`H, H, H, H, \ldots`.  But we can see this sequence is included in
:math:`A_H, A_{HH}, A_{HHH}, \ldots`.  Further, we have:

.. math::

    P(A_H) = p, P(A_{HH})=p^2, P(A_{HHH})=p^3, \ldots \tag{A.6}

so this implies the probability of :math:`P(\text{sequence of all heads}) = 0`.
This illustrates an important non-intuitive result: all sequences in our sample
space have probability :math:`0`.  Importantly, it doesn't mean they can never occur,
just that they occur "infinitesimally".  Similarly, the complement ("sequences
of at least one tails") happens with probability :math:`1`.
Mathematicians have a name for this probability :math:`1` event called *almost
surely*.  So a sequence almost surely has at least one tail.  For finite event
spaces, there is not difference between surely (always happens) and almost
surely.

This definition also includes sets of sequences that cannot be easily defined such
as:

.. math::

   \lim_{n\to \infty} \frac{H_n(\omega_1\ldots\omega_n)}{n} = \frac{1}{2} \tag{A.7}

where :math:`H_n` denotes the number of heads in the :math:`n` tosses.  This
can be implicitly constructed by taking (countably infinite) unions and intersections
of sets that we have defined in our :math:`A_\ldots` event space.  See Example
1.1.4 from [1] for more details.

Finally, although it may seem that we will have defined every subset of our
sample space, there does exist sequences that are not in
:math:`\mathcal{F}_\infty`.  But it's extremely hard to produce such a set
(and don't ask me how :p).


----

.. [1] Technically, random variables can be more general (according to Wikipedia) mapping to any measurable set.  Although, according to [1], they define it only to the real numbers.  It looks like the term `random element <https://en.wikipedia.org/wiki/Random_element>`__ is used more often for this more general case though.
