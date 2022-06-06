.. title: A Brief Introduction to Stochastic Calculus
.. slug: a-brief-introduction-to-stochastic-calculus
.. date: 2022-04-29 21:05:55 UTC-04:00
.. tags: stochastic calculus, probability, measure theory, sigma algebra, mathjax
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

:math:`\Omega` is the **sample space**, which defines the set of all possible
outcomes or results of that experiment.  In finite sample spaces, any subset of
the samples space is called an **event**.  Another way to think about events is
any thing you would want to measure the probability on, e.g. individual
elements of :math:`\Omega`,  unions of elements, or even the empty set.

However, this type of reasoning breaks down when we have certain types of
infinite samples spaces (e.g. real line).  For this, we need to define an events more precisely 
with an **event space** :math:`\mathcal{F} \subseteq 2^{\Omega}` (:math:`2^{\Omega}` denotes the 
`power set <https://en.wikipedia.org/wiki/Power_set>`__) using a construction
called a :math:`\sigma`-algebra ("sigma algebra"):

    Let :math:`\Sigma` be a non-empty set, and let :math:`\mathcal{F}` be a collection
    of subsets of :math:`\Sigma`.  We say that :math:`\mathcal{F}` is a :math:`\sigma`-`algebra <https://en.wikipedia.org/wiki/%CE%A3-algebra>`__:
    if:
    
    1. The empty set belongs to :math:`\mathcal{F}`.
    2. Whenever a set :math:`A` belongs to :math:`\mathcal{F}`, its compliment :math:`A^c` also belongs to :math:`\mathcal{F}`
       (closed under complement).
    3. Whenever a sequence of sets :math:`A_1, A_2, \ldots` belongs to :math:`\mathcal{F}`, 
       their union :math:`\cup_{n=1}^{\infty} A_n` also belongs to :math:`\mathcal{F}`
       (closed under countable unions -- implies closed under countable intersection).

    The pair :math:`(\Sigma, \mathcal{F})` define a `measurable space <https://en.wikipedia.org/wiki/Measurable_space>`__.

(NOTE: For a *very brief* discussion on countability, see Appendix A)

This sounds complicated but it basically is guaranteeing
that the subsets of :math:`\Omega` that we use for events have all the
nice properties we would expect from probabilities.  Intuitively, this helps
makes the notion of "size" or "volume" precise by defining the "chunks" of
"volume".  You want to make sure that no matter how you combine non-overlapping
"chunks" (i.e. unions of disjoint sets), you end up with a consistent measure
of "volume".  Again, this is only really needed with infinite (non-countable) sets.  For
finite event spaces, we can usually just use the power set :math:`2^{\Omega}`
as the event space, which has all these properties above.

Which brings us to our the last part of probability spaces: a **probability
measure** :math:`P` on an event space :math:`\mathcal{F}` is a function that:

1. Maps events to the unit interval :math:`[0, 1]`,
2. Returns :math:`0` for the empty set and :math:`1` for the entire space,
3. Satisfies countable additivity for all countable collections of events
   :math:`\{E_i\}` of pairwise disjoint sets:

   .. math::
 
       P(\cup_{i\in I} E_i) = \Sigma_{i\in I} P(E_i) \tag{1}

These properties should look familiar as they are the three basic ones 
axioms everyone learns when first studying probability.  The only difference is
that we're formalizing them, particularly the last one where we may not have
seen it with respect to infinite collections of events.

Going back to the "volume" analogy above, the probability measure maps the
"chunks" of our "volume" to :math:`[0,1]` (or non-negative real numbers for
general measures) but in a consistent way.  Due to the way we've defined
event spaces as :math:`\sigma`-algebra's along with the third condition from
Equation 1, we get a consistent measurement of "volume" regardless of how we
combine the "chunks".  Again, for finite sample spaces, it's not too hard to
imagine this function, but for continuous sample spaces, it gets more
complicated.  All this is essentially to define a rigorous construction that
matches our intuition of basic probability with samples spaces, events, and
probabilities.

Finally, for a given probability space :math:`(\Omega, {\mathcal {F}}, P)`,
a **random variable** :math:`X` is a `measurable function <https://en.wikipedia.org/wiki/Measurable_function>`__
:math:`X:\Omega \rightarrow E \subseteq \mathbb{R}`. 
The measurable function condition puts a few constraints:

1. :math:`X` must part of a measurable space, :math:`(E, S)` (recall:
   :math:`S` defines a :math:`\sigma`-algebra on the set :math:`E`).  
   For finite or countably infinite values of :math:`X`, we generally use
   the powerset of :math:`E`.  Otherwise, we will typically use the `Borel set
   <https://en.wikipedia.org/wiki/Borel_set>`__ for uncountably infinite
   sets (i.e. the real numbers).
2. For all :math:`S \in \mathcal{S}`, the pre-image of :math:`s` under :math:`X`
   is in :math:`\mathcal{F}`.  More precisely:

   .. math::

     \{X \in S\} := \{\omega \in \Omega | X(\omega) \in S\} \in \mathcal{F} \tag{2}

This basically says that every value that :math:`X` can take on (which must
be measurable) has a mapping to one of the measurable events
in our original event space :math:`\mathcal{F}`.  We use the notation
:math:`\sigma(X)` to denote the collection of all subsets of Equation 2,
which form the :math:`\sigma`-algebra implied by the random variable :math:`X`.

If we didn't have this condition then either: (a) we couldn't properly measure
:math:`X`'s "volume" because our "chunks" would be inconsistent (constraint 1),
or (b) we wouldn't be able to map it back to "chunks" in our original
probability space and apply :math:`P` to evaluate the random variable's
probability.  If this all seems a little abstract, it is -- that's what we need
when we're dealing with uncountable infinities.  Again, for the finite cases,
all of these properties are usually trivially met.

Using the probability measure :math:`P`, one can calculate the probability of
:math:`X \in S` using Equation 2:

.. math::

    P(X \in S) &= P(\{\omega \in \Omega | X(\omega) \in S \}) \\
               &:= P({X \in S}) \tag{3}

where :math:`S \subseteq \mathcal{S}`.  We can take :math:`S = \{x\}` to
evaluate the random variable at a particular value.  

So a random variable then allows us to map to real numbers from our original
sample space (:math:`\Omega`).  Often times our sample space has no concept
of numbers (e.g.  heads or tails) but random variables allow us to assign real
numbers to those events to calculate things like expected values and variance. 

Equation 3 basically says that we map backwards from a set of real numbers
(:math:`S`) to a set of values in the sample space (i.e. an event given by
Equation 2) using the inverse of function :math:`X`.  From the event in our
event space :math:`\mathcal{F}`, which is guaranteed to exist because of property (2),
we know how to compute the probability using :math:`P`.

For many applications of probability, understanding the above is overkill.
Most practitioners of probability can get away with the "first stage" (see box
below) of learning probability.  However specifically for stochastic calculus,
the above helps us learn it beyond a superficial level (arguably) because we
quickly get into situations where we need to understand the mathematical
rigour needed for uncountable infinities.

.. admonition:: Example 1: Sample Spaces, Events, Probability Measures, and Random Variables

   (From `Wikipedia <https://en.wikipedia.org/wiki/Event_(probability_theory)#A_simple_example>`__)

   Assume we have a standard 52 card playing deck without any jokers,
   and our experiment is that we draw a card randomly from this set.
   The sample space :math:`\Omega` is a set consisting of the 52 cards.
   An event :math:`A \subseteq \mathcal{F}` is any subset of :math:`\Omega`,
   i.e. the powerset :math:`\mathcal{F} = 2^{\Omega}`.  So that would include
   the empty set, any single element, or even the entire sample space.  Some
   examples of events:

   * "Cards that are red and black at the same time" (0 elements)
   * "The 5 of Hearts" (1 element)
   * "A King" (4 elements)
   * "A Face card" (12 elements)
   * "A card" (52 elements)

   In the case where each card is equally likely to be drawn, we 
   can define a probability measure for event :math:`A` as:
   
   .. math::

        P(A) = \frac{|A|}{|\Omega|} = \frac{|A|}{52} \tag{4}

   We can additionally define a random variable as:
   
   .. math::

        X(\omega \in \Omega) = 
        \begin{cases}
            1 &\text{if } \omega \text{ is red}\\
            0 &\text{otherwise}
        \end{cases}
        \tag{5}

   Which is a mapping from our sample space :math:`\Omega` to a (finite) subset
   of the real numbers :math:`\{0, 1\}`.  We can calculate probabilities using
   Equation 3, for example :math:`X = 1`:

   .. math::
        
        P(X \in \{1\}) &= P(\{\omega \in \Omega | X(\omega) \in \{1\} \}) \\
        &= P(\{\omega | \omega \text{ is a red card}\}) \\
        &= \frac{|\{\text{all red cards}\}|}{52} \\
        &= \frac{1}{2}  \\
        \tag{6}

   The implied :math:`\sigma`-algebra of this random variable can be defined as:
   :math:`\sigma(X) = \{ \emptyset, \text{"all red cards"}, \text{"all black cards"}, \Omega \} \subset \mathcal{F}`.

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
    to a subset of the real numbers :math:`\mathbb{R}`.  Certain subsets
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


Stochastic Processes
--------------------

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
experiments performed at each index in :math:`T`. (Note: by definition it's infinite
because otherwise it would just be a random vector.)  For example, flipping a 
coin at every (discrete) time from :math:`0` to :math:`\infty`, would define a
specific infinite sequence of heads and tails :math:`\omega = \{H, T, H, H, H, T, \ldots\}`.
So each random variable :math:`X_t` can depend on the entire sequence of the
outcome of this infinite "experiment".  That is, :math:`X_t` is a mapping
from outcomes of our infinite experiment to (a subset of) the real numbers: 
:math:`X_t: \Omega \rightarrow E \subseteq \mathbb{R}`.
It's important to note that in this general definition we have no explicit
concept of time, so we can depend on the "future".  To include our usual
concept of time, we need an additional concept (see adapted processes below).

Finally, instead of viewing the stochastic process as a collection of random variables
indexed by time, we could look at it as a function of both time and the sample space
i.e., :math:`X(t, \omega) = X_t(\omega)`.  For a given outcome of an experiment
:math:`\omega_0`, the deterministic function generated as :math:`X(t, \omega=\omega_0)` is
called the **sample function**.  However, mostly we like to think of it
as having a random variable at each time step indicated by this notation: 
:math:`X = \{X_t, t \in T\}`.  We sometimes use the notation :math:`X(t)` to refer
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
    \in \{H, T\} \}`, that is, the set of all infinite sequences of "heads" and "tails".
    It turns out the event space and the probability measure are surprisingly
    complex to define so I've put those details in Appendix A.

    We can define the random variable given an outcome of infinite tosses
    :math:`\omega`:

    .. math::

        X_t(\omega) =  \begin{cases}
            1 &\text{if } \omega_t = H\\
            0 &\text{otherwise}
        \end{cases} \tag{7}

    for :math:`\omega = \omega_1 \omega_2 \omega_3 \ldots`, where each :math:`\omega_i`
    is the outcome of the :math:`i^{th}` toss.
    For all values of :math:`t`, the probability :math:`P(X_t = 1) = p`, for
    some constant :math:`p \in [0, 1]`.

.. admonition:: Example 3: One Dimensional Symmetric Random Walk

   A simple one dimensional symmetric `random walk <https://en.wikipedia.org/wiki/Random_walk>`__
   is a discrete value, discrete time stochastic process.  An easy way to 
   think of it is: starting at 0, at each time step, flip a fair coin and move
   right (+1) if heads, otherwise move left (-1).

   This can be defined in terms of the Bernoulli process :math:`X_t` from
   Example 2 with :math:`p=0.5` (with the same probability space):

   .. math::

        S_t(\omega) =  \sum_{i=1}^t X_t \tag{8}

   Notice that the random variable at each time step depends on *all* the "coin
   flips" :math:`X_t` that came before it in contrast to just the current "coin flip"
   for the Bernoulli process.
   
   Another couple of results that we'll use later.  First is that the increments
   between any two given non-overlapping pairs of integers
   :math:`0 = k_0 < k_1 < k_2 < \ldots < k_m` are independent.  That is,
   :math:`(S_{k_1} - S_{k_0}), (S_{k_2} - S_{k_1}), (S_{k_3} - S_{k_2}), \ldots, (S_{k_m} - S_{k_{m-1}})`
   are independent.  We can see this because for any combination of pairs of
   these differences, we see that the independent :math:`X_t` variables don't
   overlap, so the sum of them must also be independent.

   Moreover, the expected value and variance of the differences is given by:
   
   .. math::

        E[S_{k_{i+1}} - S_{k_i}] &= E[\sum_{j=k_i + 1}^{k_{i+1}} X_i] \\
                                 &= \sum_{j=k_i + 1}^{k_{i+1}} E[X_j] \\
                                 &= 0 \\
        Var[S_{k_{i+1}} - S_{k_i}] &= E[\sum_{j=k_i + 1}^{k_{i+1}} X_i] \\
                                   &= \sum_{j=k_i + 1}^{k_{i+1}} Var[X_j]  && X_i \text{ independent}\\
                                   &= \sum_{j=k_i + 1}^{k_{i+1}} 1 && Var[X_j] = E[X_j^2] = 1 \\
                                   &= k_{i+1} - k_i \\
        \tag{9}

   Which means that the variance of the symmetric random walk accumulates
   at a rate of one per unit time.  So if you take :math:`l` steps from the
   current position, you can expect a variance of :math:`l`.  We'll see this
   pattern when we discuss the extension to continuous time.


Adapted Processes
-----------------

Notice that in the previous section, our definition of stochastic process
included a random variable :math:`X_t: \Omega \rightarrow E \subseteq \mathbb{R}`
where each :math:`\omega \in \Omega` is an infinite set representing a
given outcome for the infinitely long experiment.  This implicitly means
that at "time" :math:`t`, we could depend on the "future" because we are
allowed to depend on any tosses, including those greater than :math:`t`.  In
many applications, we do want to interpret :math:`t` as time so we wish to
restrict our definition of stochastic processes.

An `adapted stochastic process <https://en.wikipedia.org/wiki/Adapted_process>`__
is one that cannot "see into the future".  Informally, it means that for
any :math:`X_t`, you can determine it's value by *only* seeing the outcome 
of the experiment up to time :math:`t` (i.e., :math:`\omega_1\omega_2\ldots\omega_t` only).

To define this more formally, we need to introduce a few technical definitions
to define this fully.  We've already seen the definition of the
:math:`\sigma`-algebra :math:`\sigma(X)` implied by the random variable
:math:`X` in a previous subsections.  Suppose we have a subset of our event
space :math:`\mathcal{G}`, we say that :math:`X` is
:math:`\mathcal{G}`-measurable if every set in :math:`\sigma(X) \subseteq \mathcal{G}`.
That is, we can use :math:`\mathcal{G}` to "measure" anything we do with :math:`X`.

Using this idea, we define the concept of a filtration
on our event space :math:`\mathcal{F}` and our index set :math:`T`:

    A **filtration** :math:`\mathbb{F}` is a ordered collection
    of subsets :math:`\mathbb{F} := (\mathcal{F_t})_{t\in T}` where 
    :math:`\mathcal{F_t}` is a sub-:math:`\sigma`-algebra of :math:`\mathcal{F}`
    and :math:`\mathcal{F_{t_1}} \subseteq \mathcal{F_{t_2}}` for all
    :math:`t_1 \leq t_2`.

To break this down, we're basically saying that our event space :math:`\mathcal{F}`
can be broken down into logical "sub event spaces" :math:`\mathcal{F_t}` such
that each one is a superset of the next one.  This is precisely what we want
where as we progress through time, we "gain" more "information" but never lose
any.  We can also use this idea of defining a sub-:math:`\sigma`-algebra to
formally define conditional probabilities, although we won't cover it in this
post (see [1] for a more detailed treatment).

Using the construct of a filtration, we can define:

    A stochastic process :math:`X_t : T \times \Omega` that is **adapted to the
    filtration** :math:`(\mathcal{F_t})_{t\in T}` if the random variable
    :math:`X_t` is :math:`F_t`-measurable. 
   
This basically says that :math:`X_t` can only depend on "information" before or
at time :math:`t`.  The "information" available is encapsulated by the
:math:`\mathcal{F_t}` subsets of the event space.  These subsets of events are
the only ones we can compute probabilities on for that particular random
variable, thus effectively restricting the "information" we can use.
As with much of this topic, we require a lot of rigour in order to make sure we
don't have weird corner cases.  The next example gives more intuition on
the interplay between filtrations and random variables.

.. admonition:: Example 4: An Adapted Bernoulli Processes

    First, we need to define the filtration that we wish to adapt to our
    Bernoulli Process.  Borrowing from Appendix A, repeating the two equations:

    .. math::

        A_H &= \text{the set of all sequences beginning with } H = \{\omega: \omega_1 = H\} \\
        A_T &= \text{the set of all sequences beginning with } T = \{\omega: \omega_1 = T\} \\
        \tag{10}
 
    This basically defines two events (i.e., sets of infinite coin toss
    sequences) that we use to define our probability measure.  We define our
    first sub-:math:`\sigma`-algebra using these two sets:

    .. math::

        \mathcal{F}_1 = \{\emptyset, \Sigma, A_H, A_T\} \tag{11}

    Let's notice that :math:`\mathcal{F}_1 \subset \mathcal{F}` (by definition
    since this is how we defined it). Also let's take a look at the events generated
    by the random variable for heads and tails:

    .. math::

           \{X_1 \in \{H\}\} &= \{\omega \in \Sigma | X_1(omega) \in {H}\} \\
            &= \{\omega: \omega_1 = H\} \\
            &= A_H \\
           \{X_1 \in \{H\}\} &= \{\omega \in \Sigma | X_1(omega) \in {T}\} \\
            &= \{\omega: \omega_1 = T\} \\
            &= A_T \\
            \tag{12}

    Thus, :math:`\sigma(X_1) = \mathcal{F}_1` (the :math:`\sigma`-algebra implied by
    the random variable :math:`X_1`, meaning that :math:`X_1` is indeed
    :math:`\mathcal{F}_1`-measurable as required.  
    
    Let's take a closer look at what this means.  For :math:`X_1`, Equation 11 defines 
    the only types of events we can measure probability on, in plain English:
    empty set, every possible outcome, outcomes starting with the first coin
    flip as heads, and outcomes starting with the first coin flip as tails.
    This corresponds to probabilities of :math:`0, 1, p` and :math:`1-p`
    respectively, precisely the outcomes we would expect :math:`X_1` to be able
    to calculate with :math:`X_1`.
    
    On closer examination though, this is not exactly the same as a naive understanding
    of the situation would imply.  :math:`A_H` contains *every* infinitely long
    sequence starting with heads -- not just the result of the first flip.
    Recall, each "time-indexed random variable in a stochastic process is a
    function of an element of our sample space, which is an infinitely long sequence.
    So we cannot naively pull out just the result of the first toss.  Instead, we
    group all sequences that match our criteria (heads on the first toss) together
    and use that as a grouping to perform our probability "measurement" on.  Again,
    it may seem overly complicated but this rigour is needed to ensure we don't
    run into weird problems with infinities.
  
    Continuing on for later "times", we can define :math:`\mathcal{F}_2,
    \mathcal{F}_3, \ldots` and so on in a similar manner. We'll find that each
    :math:`X_t` is indeed :math:`\mathcal{F}_t` measurable (see Appendix A for
    more details), and also find that each one is a superset of its
    predecessor.  As a result, we can say that the Bernoulli process
    :math:`X(t)` is adapted to the filtration :math:`(\mathcal{F_t})_{t\in
    \mathbb{N}}` as defined in Appendix A.
    
Brownian Motion
---------------

`Brownian motion <https://en.wikipedia.org/wiki/Wiener_process>`__ (also known as
the Weiner process) is one of the most widely studied continuous time
stochastic processes.  It occurs frequently in many different domains such as
applied math, quantitative finance, and physics.  As alluded to previously, it
has many "corner case" properties that do not allow simple manipulation, and
it is one of the reasons why stochastic calculus was discovered.
Interestingly, there are several equivalent definitions but we'll start with
the one defined in [1] using scaled random walks.


Scaled Symemtric Random Walk
****************************

A scaled symmetric random walk process is an extension of the simple random
walk we showed in Example 3 except that we "speed up time and scale down the
step size" and extend it to continuous time.  More precisely, for a fixed
positive integer :math:`n`, we define the scaled random walk as:

.. math::

    W^{(n)}(t) = \frac{1}{\sqrt{n}}S_{nt} \tag{13}

where :math:`S_{nt}` is a simple symmetric random walk process, provided that
:math:`nt` is an integer.  If :math:`nt` is not an integer, we'll simply define
:math:`W^{(n)}(t)` as the linear interpolation between it's nearest integer
values.  

A simple way to think about Equation 13 is that it's just a regular random walk
with a scaling factor.  For example, :math:`W^{(100)}(t)` has it's first step
(integer step) at :math:`t=\frac{1}{100}` instead of :math:`t=1`.  To adjust
for this compression of time we scale the process by :math:`\frac{1}{\sqrt{n}}`
to make the math work out later.  The linear interpolation is not that relevant
except that we want to start working in continuous time.

Since this is just a simple symmetric random walk (assuming we're analyzing
it as its integer steps), the same properties hold as we discussed in Example
3.  Namely, that non-overlapping increments are independent.  Additionally, for
:math:`0 \leq s \leq t`, we have:

.. math::

    E[W^{(n)}(t) - W^{(n)}(s)] &= 0 \\
    Var[W^{(n)}(t) - W^{(n)}(s)] &= t - s \\
    \tag{14}

where we use the square root scaling to end up with variance accumulating still
at one unit per time. 

Another property, we'll look at is a quantity called the 
`quadratic variation <https://en.wikipedia.org/wiki/Quadratic_variation>`__,
which is calculated *along a specific path* (i.e., there's not randomness
involved).  For a scaled symmetric random walk, we get:

.. math::

    [W^{(n)}, W^{(n)}]_t &= \sum_{j=1}^{nt} (W^{(n)}(\frac{j}{n} - W^{(n)}(\frac{j-1}{n}))^2 \\
    &= \sum_{j=1}^{nt} [\frac{1}{\sqrt{n}} X_j]^2  \\
    &= \sum_{j=1}^{nt} \frac{1}{n} = t \\
    \tag{15}

This results in the same quantity as the variance computation we have (for
:math:`s=0`) in Equation 14 but is conceptually different.  The variance
is an average over all paths, while the quadratic variation is taking a
realized path, squaring all the values, and then summing them up.
Interestingly, they result in the same thing.

Finally, as you might expect, we wish to understand what happens
to the scaled symmetric random walk when :math:`n \to \infty`.
For a given :math:`t\geq 0`, let's recall a few things:

* :math:`E[W^{(n)}(t)] = 0` (from Equation 14 with :math:`s = 0`).
* :math:`Var[W^{(n)}(t)] = t` (from Equation 14 with :math:`s = 0`).
* :math:`W^{(n)}(t) = \frac{1}{\sqrt{n}} \sum_{i=1}^t X_t` for Bernoulli process :math:`X(t)`.
* The `central limit theorem <https://en.wikipedia.org/wiki/Central_limit_theorem#Classical_CLT>`__
  states that :math:`\frac{1}{\sqrt{N}}\sum_{i=1}^n Y_i` converges
  to :math:`\mathcal{N}(\mu_Y, \sigma_Y^2)` as :math:`n \to \infty` for IID
  random variables :math:`Y_i` (given some mild conditions).

We can see that our symmetric scaled random walk fits precisely the conditions
as the central limit theorem, which means that as :math:`n \to \infty`,
:math:`W^{(n)}(t)` converges to a normal distribution with mean :math:`0` and
variance :math:`t`.  This limit is in fact the method in which we'll define
Brownian motion.

Brownian Motion Definition
**************************

We finally arrive at the definition of Brownian motion, which will be the limit
of the scaled symmetric random walk as :math:`n \to \infty`.  We'll define it
in terms of the properties of this limiting distribution, many of which are inherited
from the scaled symmetric random walk:

    Given probability space :math:`(\Sigma, \mathcal{F}, P)`,
    For each :math:`\omega \in Omega`, define a continuous function that depends on
    :math:`\omega` as :math:`W(t) := W(t, \omega)` for :math:`t \geq 0`.
    :math:`W(t)` is a **Brownian motion** if the following are satisfied:

    1. :math:`W(0) = 0`;
    2. All increments :math:`W(t_1) - W(t_0), \ldots, W(t_m) - W(t_{m-1})`
       for :math:`0 = t_0 < t_1 < \ldots < t_{m-1} < t_{m}` are independent; and
    3. Each increment is distributed normally with :math:`E[W(t_{i+1} - t_i)] = 0` and 
       :math:`Var[W(t_{i+1} - t_i)] = t_{i+1} - t_i`.

We can see that Brownian motion inherits many of the same properties as our scaled
symmetric random walk.  Namely, independent increments with each one being
distributed normally.  With Brownian motion the increments are exactly normal
instead of approximately normal (for large :math:`n`) with the scaled symmetric
random walk.

One way to think of Brownian motion is that each :math:`\omega` is a path generated
by a random experiment, for example, the random motion of a particle suspended
in a fluid.  At each infinitesimal point in time, it is perturbed randomly
(distributed normally) into a different direction.  In fact, this is the origin
of the phenomenon by botanist `Robert Brown
<https://en.wikipedia.org/wiki/Robert_Brown_(botanist,_born_1773)>`__ 
(although the math describing it came after by several others including Einstein).

Another way to think about the random motion is using our analogy of coin tosses.
:math:`\omega` is still the outcome of an infinite sequence of coin tosses but
instead of happening at each integer value of :math:`t`, they are happening
"infinitely fast".  This is essentially the result of taking our limit to infinity.

We can ask any questions that we usually would ask about random variables with
Brown motion.  The next example shows a few of them.

.. admonition:: Example 5: Brownian Motion

    Suppose we wish to determine the probability that Brownian motion
    at :math:`t=0.25` is between :math:`0` and :math:`0.25`.  Using
    our rigourous jargon, we would say that we want to determine
    the probability of the set :math:`A \in \mathcal{F}` containing
    :math:`\omega \in \Omega` satisfying :math:`0 \leq W(0.25) \leq 0.2`.

    We know that each increment is normally distributed with expectation of
    :math:`0` and variance of :math:`t_{i+1}-t_{i}`, so for the :math:`[0, 0.25]`
    increment, we have:

    .. math::

        W(0.25) - W(0) = W(0.25) - 0 = W(0.25) \sim N(0, 0.25) \tag{16}

    Thus, we are just asking the probability that a normal distribution takes
    on these values, which we can easily compute using the normal distribution density:

    .. math::

        P(0 \leq W(0.25) \leq 0.2) &= \frac{1}{\sqrt{2\pi(0.25)}} \int_0^{0.2} e^{-\frac{1}{2}(\frac{x}{0.25})^2}  \\
                                   &= \frac{2}{2\pi} \int_0^{0.2} e^{-2x^2}  \\
                                   &\approx 0.155 \\
                                   \tag{17}

We also have the concept of filtrations for Brownian motion.  It uses the same definition
as we discussed previously except it also adds the condition that future increments
are independent of any :math:`\mathcal{F_t}`.  As we will see below, we will be
using more complex adapted stochastic processes as integrands against a Brownian
motion integrator.  This is why it's important to add this additional condition
of independence for future increments.  It's so the adapted stochastic process
(with respect to the Brownian motion filtration) can be properly integrated
and cannot "see into the future".


Quadratic Variation of Brownian Motion
**************************************

We looked at the quadratic variation above for the scaled symmetric random walk
and concluded that it accumulates quadratic variation one unit per time (i.e.
quadratic variation is :math:`T` for :math:`[0, T]`) regardless of the value of
:math:`n`.  We'll see that this is also true for Brownian motion but before we
do, let's first appreciate why this is strange.

    Let :math:`f(t)` be a function defined on :math:`[0, T]`.  The 
    **quadratic variation** of :math:`f` up to :math:`T` is

    .. math::

        [f, f](T) = \lim_{||\Pi|| \to 0} \sum_{j=0}^{n-1}[f(t_{j+1}) - f(t_j)]^2 \tag{18}

    for :math:`\Pi = \{t_0, t_1, \ldots, t_n\}`, :math:`0\leq t_1 \leq t_2 < \ldots < t_n = T`
    and :math:`||\Pi|| = \max_{j=0,\ldots,n} (t_{j+1}-t_j)`.

This is basically the same idea that we discussed before: for infinitesimally
small intervals, take the difference of the function for each interval,
square them, and then sum them all up.  The part you may not be familiar with
is that instead of having an evenly spaced intervals like we usually see in a
first calculus course, we're can use any unevenly spaced ones.  The only 
condition is that the largest partition goes to zero.  This is called the mesh
or norm of the partition, which is similar to the formal definition of 
`Riemannian integrals <https://en.wikipedia.org/wiki/Riemann_integral>`__
(even though many of us, like myself, didn't learn it this way).  In any
case the idea is very similar to just having evenly spaced intervals.

Now that we have Equation 18, let's see how it behaves on a function
:math:`f(t)` that has a continuous derivative:
(recall the `mean value theorem <https://en.wikipedia.org/wiki/Mean_value_theorem>`__ 
states that :math:`f'(c) = \frac{f(a) - f(b)}{b-a}` for :math:`c \in (a,b)`
for continuous functions with derivatives on the respective interval):

    .. math::

        [f, f](T) &= \lim_{||\Pi|| \to 0} \sum_{j=0}^{n-1}[f(t_{j+1}) - f(t_j)]^2   && \text{definition} \\
        &= \lim_{||\Pi|| \to 0} \sum_{j=0}^{n-1}|f'(t_j^*)|^2 (t_{j+1} - t_j)^2 && \text{mean value theorem} \\
        &\leq \lim_{||\Pi|| \to 0} ||\Pi|| \sum_{j=0}^{n-1}|f'(t_j^*)|^2 (t_{j+1} - t_j)  \\
        &= \big[\lim_{||\Pi|| \to 0} ||\Pi||\big] \big[\lim_{||\Pi|| \to 0} \sum_{j=0}^{n-1}|f'(t_j^*)|^2 (t_{j+1} - t_j)\big] && \text{limit product rule}  \\
        &= \big[\lim_{||\Pi|| \to 0} ||\Pi||\big] \int_0^T |f'(t)|^2 dt = 0&& f'(t) \text{ is continuous} \\
        \tag{19}

So we can see that quadratic variation is not very important for most functions
we are used to seeing i.e., ones with continuous derivatives.  In cases where
this is not true, we cannot use the mean value theorem to simplify quadratic
variation, so we potentially will get something that is non-zero.

For Brownian motion in particular, we do not have a continuous derivative
and cannot use the mean value theorem as in Equation 19, so we end up with
a non-zero quadratic variation.  To see this, let's take a look at the absolute
value function :math:`f(t) = |t|` in Figure 1.  On the interval :math:`(-2, 5)`,
the slope between the two points is :math:`\frac{3}{7}`, but nowhere in this
interval is the slope of the absolute value function :math:`\frac{3}{7}` (it's
either constant 1 or constant -1 or undefined).

.. figure:: /images/stochastic_calculus_mvt.png
    :width: 500px
    :alt: Mean value theorem does not apply on functions without derivatives
    :align: center

**Figure 1: Mean value theorem does not apply on functions without derivatives (`source <https://people.math.sc.edu/meade/Bb-CalcI-WMI/Unit3/HTML-GIF/MeanValueTheorem.html>`__)**

Recall, this is a similar situation to what we had for the scaled symmetric 
random walk -- in between each of the discrete points, we used a linear
interpolation.  As we increase :math:`n`, this "pointy" behaviour persists and
is inherited by Brownian motion where we no longer have a continuous
derivative.  Thus, we need to deal with this situation where we have a function
that is continuous everywhere, but differentiable nowhere.  This is one of the
key reasons why we need stochastic calculus, otherwise we could just use the
rules for standard calculus we all know and love.

.. admonition:: **Theorem 1** For Brownian motion :math:`W`, :math:`[W,W](T) = T`
    for all :math:`T\geq 0` almost surely.

    **Proof**

    Define the sampled quadratic variation for partition as above (Equation 18):

    .. math::

        Q_{\Pi} = \sum_{j=0}^{n-1}\big( W(t_{j+1}) - W(t_j) \big)^2 \tag{20}

    This quantity is a random variable since it depends on the particular
    "outcome" path of Brownian motion (recall quadratic variation is with
    respect to a particular realized path).  
    
    To prove the theorem, We need to show that the sampled quadratic variation
    converges to :math:`T` as :math:`||\Pi|| \to 0`.  This can be accomplished
    by showing :math:`E[Q_{\Pi}] = T` and :math:`Var[Q_{\Pi}] = 0`, which says
    that we will converge to :math:`T` regardless of the path taken.

    We know that each increment in Brownian motion is independent, thus
    their sums are the sums of the respective means and variances of each
    increment.  So given that we have:

    .. math::

        E[(W(t_{j+1})-W(t_j))^2] &= E[(W(t_{j+1})-W(t_j))^2] - 0 \\
                                 &= E[(W(t_{j+1})-W(t_j))^2] - E[W(t_{j+1})-W(t_j)]^2 && \text{definition of Brownian motion}\\
                                 &= Var[W(t_{j+1})-W(t_j)]  \\
                                 &= t_{j+1} -  t_j && \text{definition of Brownian motion}\\
                                 \tag{21}

    We can easily compute :math:`E[Q_{\Pi}]` as desired:

    .. math::

        E[Q_{\Pi}] &= E[ \sum_{j=0}^{n-1}\big( W(t_{j+1}) - W(t_j) \big)^2 ] \\
        &= \sum_{j=0}^{n-1} E[W(t_{j+1}) - W(t_j)]^2 \\
        &= \sum_{j=0}^{n-1} (t_{j+1} - t_j)  && \text{Equation } 21 \\
        &= T \\
        \tag{22}

    From here, we use the fact <https://math.stackexchange.com/questions/1917647/proving-ex4-3%CF%834>`__ 
    that the expected value of the fourth moment of a normal random variable
    with zero mean is three times its variance.  Anticipating the quantity
    we'll need to compute the variance, we have:

    .. math::

         E\big[(W(t_{j+1})-W(t_j))^4 \big] = 3Var[(W(t_{j+1})-W(t_j)] = 3(t_{j+1} - t_j)^2 \tag{23}

    Computing the variance of each increment:

    .. math::
    
         Var\big[(W(t_{j+1})-W(t_j))^2 \big] &= E\big[\big( (W(t_{j+1})-W(t_j))^2 -  E[(W(t_{j+1})-W(t_j))^2] \big)\big] && \text{definition of variance} \\
         &= E\big[\big( (W(t_{j+1})-W(t_j))^2 -  (t_{j+1} - t_j) \big)\big] && \text{Equation } 21 \\
         &= E[(W(t_{j+1})-W(t_j))^4] - 2(t_{j+1}-t_j)E[(W(t_{j+1})-W(t_j))^2] + (t_{j+1} - t_j)^2 \\
         &= 3(t_{j+1}-t_j)^2 - 2(t_{j+1}-t_j)^2 + (t_{j+1} - t_j)^2 && \text{Equation } 21/23 \\
         &= 2(t_{j+1}-t_j)^2 \\
         \tag{24}

    From here, we can finally compute the variance:

    .. math::

        Var[Q_\Pi] &= \sum_{j=0}^{n-1} Var\big[ (W(t_{j+1} - W(t_j)))^2 \big]  \\
                   &= \sum_{j=0}^{n-1} 2(t_{j+1}-t_j)^2  && \text{Equation } 24 \\
                   &\leq  \sum_{j=0}^{n-1} 2 ||\Pi|| (t_{j+1}-t_j)  \\
                   &= 2 ||\Pi|| T && \text{Equation } 22 \\
                   \tag{25}

    As :math:`\lim_{||\Pi|| \to 0} Var[Q_\Pi] = 0`, therefore we have shown that
    :math:`\lim_{||\Pi|| \to 0} Q_\Pi = T` as required.

* TODO: Give some summary of what this means.  Talk about almost surely.
* Talk about shorthand dWdW = dt
* accumulates QV at one unit per time
* Mention cross variation is 0, dWdt = 0 and dtdt = 0
 

* Example (use something from Hull textbook)

First Passage of Time for Brownian Motion
*****************************************

An interesting question to ask is the *first passage of time* question: 
* First passage of time is almost surely finite
* Surely p=1.0 to return to value, and it's length is infinite

Stochastic Integrals
====================

* Adapted Processes: https://en.wikipedia.org/wiki/Adapted_process
  * Itō integral, which only makes sense if the integrand is an adapted process. 


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

