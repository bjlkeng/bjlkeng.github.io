.. title: The Logic Behind the Maximum Entropy Principle
.. slug: the-logic-behind-entropy
.. date: 2024-07-03 20:44:59 UTC-04:00
.. tags: entropy, information, Shannon, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

For a while now, I've really enjoyed diving deep to understand
probability and related fundamentals (see 
`here <link://slug/probability-the-logic-of-science>`__,
`here <link://slug/maximum-entropy-distributions>`__, and
`here <link://slug/an-introduction-to-stochastic-calculus>`__).
Entropy is a topic that comes up all over the place from physics to information
theory, and of course, machine learning.  I written about it in various
different forms but always taken it as a given as the "expected information".
Well I found a few of good explanations about how to "derive" it and thought
that I should share.

In this post, I'll be showing a few of derivations of the maximum entropy
principle, where entropy appears as part of the definition.  These derivations
will show why it is a reasonable and natural thing to maximize, and how it is
determined from some well thought out reasoning.  This post will be more math
heavy but hopefully it will give you more insight into this wonderfully
surprising topic.


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

To understand entropy, let's first concoct a situation where we might need a new
tool beyond `Bayesian probability <https://en.wikipedia.org/wiki/Bayesian_probability>`__.
Suppose we have a die that has faces from 1 to 6 where we define the random
variable :math:`X` to be equal to the face-up number from a given die roll.  If we are given samples of die rolls
then, with some appropriate prior, we can iteratively apply Bayes' theorem to
determine the probability :math:`P(X=i) = p_i` where :math:`i` is die face value.
This type of calculation is relatively straight forward, maybe tedious, but straight
forward.  But what if we don't have explicit samples but a different type of observable
information?

Now imagine we don't have samples of that die roll.  Instead we only know its mean
roll, that is :math:`E[X] = \mu_X`, what would our best guess be of the probability
distribution of :math:`X`?  For example, if all the die faces had
equal probability we would get:

.. math::

   E[X] = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = 3.5 \tag{1}

So if :math:`\mu_X = 3.5` then we might reasonably guess that :math:`X` had a
uniform distribution.  

But what if :math:`\mu_X = 3`?  It could be that :math:`p_3=1.0`, but that
seems unlikely.  Maybe something like this is slightly more reasonable?

.. math:: 

   {\bf p} &= [0.2, 0.2, 0.2, 0.2, 0.2, 0] \\
   E[X] &= 0.2(1 + 2 + 3 + 4 + 5) + 0(6) = 3 \tag{2}

But still that zero probability for :math:`p_6` seems kind of off.
One way to approach it is to apply a prior to each :math:`p_i` and
marginalizing over all of them but ensuring it matches our information of
:math:`E[X] = \mu_X`.  But that only shifts the problem to finding the right
priors to match our observation, not how to directly incorporate our
information of :math:`E[X] = \mu_X`.

From this example, we have gleaned some important insights though.
Somehow concentrating all the mass together with :math:`p_3=1.0` is not exactly
what we want, and while Equation 2 might seem slightly more reasonable,
it also seems odd that we have :math:`p_6=0`. This gives us intuition that
we want to be conservative and "spread out" our probability, and at the same
time assigning an outcome with :math:`0` probability only if it is truly ruled
out.  This spreading allows us to be noncommittal about the distribution.  So
in some sense we want to maximize the spread given the constraint, which
already sounds like something `familiar
<link://slug/maximum-entropy-distributions>`__ perhaps?  Let's keep going and
find out.

Entropy and The Principle of Maximum Entropy
============================================

Information theoretic `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`__
is typically interpreted as the average amount of "information", "surprise" or "uncertainty"
in a given random variable.  Given a random variable :math:`X` that has :math:`m`
discrete outcomes, we can define entropy as:

.. math::

   H(X) = -\sum_{x \in X} p_x \log p_x \tag{3}

where :math:`p_x := P(X=x)`.  

The principle of maximum entropy states that given testable information,
the probability distribution that best represents the current state of
knowledge is the one with the largest (information) entropy. 
The *testable information* is a statement about the probability distribution
where you can easily evaluate it to be true or false. It usually comes in the
form of a mean, variance, or some moments of the probability distribution.
This rule can be thought of expressing epistemic modesty, or maximal ignorance,
because it makes the least strong claim on a distribution beyond being informed
by the testable information.

See my previous post on `maximum entropy distributions
<link://slug/maximum-entropy-distributions>`__ for a more detailed
treatment.


Wallis Derivation
=================

The Wallis derivation of maximum entropy [1 ,2] starts off with a conceptually
simple argument, which through some calculation ends up with our tried and true
measure of entropy.  It's nice because it does not require defining information
as :math:`-\log p` upfront and then making some argument that we want to
maximize it's expected value but gets there directly instead.

To start, let's assume the problem from the previous section: we want to
find the probability distribution of a discrete random variable that has
:math:`m` discrete values given some testable information about the
distribution such as it's expected value or variance (or more generally one of
its `moments <https://en.wikipedia.org/wiki/Moment_(mathematics)>`__).  Given
any candidate probability distribution represented by a vector 
:math:`{\bf p} = [p_1, \ldots, p_m]`, we can easily test to see if it satisfies
our constraint.  To this end, let's conduct the following thought experiment:

1. Distribute :math:`N` quanta of probability (each worth :math:`\frac{1}{N}`)
   *uniformly randomly* across the :math:`m` possibilities for some large :math:`N`.
2. Once finished, check if the constraint is satisfied.  If so, then that is 
   your desired probability assignment.  If not reject and go back to step 1.

If we do get an acceptance, our distribution will have :math:`p_i =
\frac{n_i}{N}` where :math:`\sum_{i=1}^m n_i = N`.  Note: for any reasonably
large :math:`N` and skewed distribution, it will take an astronomically large
number of iterations to accept, but it is only a thought experiment.

Now why is this a reasonable way to approach the problem?  First, we're
*uniformly* randomly distributing our quanta of probability in step 1.  It's
hard to argue that we're being biased in any way.  Second, if we pick a
large enough :math:`N`, the chances of getting a "weird" probability
distribution (like the :math:`p_3=1.0` from the previous section) over a more
reasonable one becomes vanishing small.  So even though we're stopping at the
first one, chances are it's a pretty reasonable distribution.

Assuming we're happy with that reasoning, there is still the problem of picking
a large enough :math:`N` and running many iterations in order to get to an
accepted distribution.  Instead, let's just calculate the most probable result
from this experiment, which should be the most reasonable choice anyways.
We can see the probability of any particular assignment of our probability quanta
is a `multinomial distribution <https://en.wikipedia.org/wiki/Multinomial_distribution>`__
with the probability of a quanta being assigned to an outcome being a constant :math:`q_i = \frac{1}{m}`:

.. math::

   P({\bf q}) &= \frac{N!}{n_1!\ldots n_m!}q_1^{n_1}q_2^{n_2} \ldots q_m^{n_m} \\
   &= \frac{N!}{n_1!\ldots n_m!}m^{-N} 
   &&& \text{since } q_i = \frac{1}{m} \text{ and } \sum_{i=1}^m n_i = N\\
   \tag{4}

since :math:`m` is a constant in this problem, it suffices to maximize the first factor, which
we'll call the **multiplicity** of the outcome denoted by :math:`W`:

.. math::

   W = \frac{N!}{n_1!\ldots n_m!} \tag{5}

But we can equivalently maximize any monotonically increasing function of :math:`W`,
so let's try it with :math:`\frac{1}{N}\log W`:

.. math::

   \frac{1}{N} \log W &= \frac{1}{N} \log \frac{N!}{n_1!n_2!\ldots n_m!} \\
   &= \frac{1}{N} \log \frac{N!}{(N\cdot p_1)!(N\cdot p_2)!\ldots (N\cdot p_3)!} \\
   &= \frac{1}{N} \Big( \log N! - \sum_{i=1}^m \log((N\cdot p_i)!) \Big) \\
   \tag{6}

The factorials in Equation 6 are annoying to deal with but thankfully we can use 
`Sterling's approximation <https://en.wikipedia.org/wiki/Stirling%27s_approximation>`__:

.. math::

   \log(n!) = n\log n -n + \mathcal{O}(\log n) \tag{7}

With Equation 7 in hand, we can simplify Equation 6 and take the limit as :math:`N \to \infty`
so we reduce our dependence on finite :math:`N`:

.. math::

   \lim_{N\to\infty} \frac{1}{N} \log W 
   &= \lim_{N\to\infty} \frac{1}{N} \Big( \log N! - \sum_{i=1}^m \log((N\cdot p_i)!) \Big) \\
   &= \lim_{N\to\infty} \frac{1}{N} \Big( N\log N - n + \mathcal{O}(\log N) 
       && \text{Sterling's approx.}\\
   &\hspace{4.5em} - \sum_{i=1}^m (N\cdot p_i)\log(N\cdot p_i) - (N\cdot p_i) + \mathcal{O}(\log (N\cdot p_i)) \Big) \\
   &= \lim_{N\to\infty} \frac{1}{N} \Big( N\log N 
    - \sum_{i=1}^m (N\cdot p_i)\log(N\cdot p_i)  \Big) && \text{Drop lower order terms} \\
   &= \lim_{N\to\infty} \log N - \sum_{i=1}^m p_i\log(N\cdot p_i)   \\
   &= \lim_{N\to\infty} \log N - \log N \sum_{i=1}^m p_i - \sum_{i=1}^m p_i\log p_i   \\
   &= \lim_{N\to\infty} \log N - \log N - \sum_{i=1}^m p_i\log p_i   \\
   &= \lim_{N\to\infty} - \sum_{i=1}^m p_i\log p_i \\
   &= - \sum_{i=1}^m p_i\log p_i \\
   &= H({\bf p}) \\
   \tag{8}

Equation 8 shows that if we follow the logic of the above procedure, the "fair"
probability distribution is equivalent to maximizing the entropy of the
distribution.  Notice that we did not mention "information", "surprise", or
"uncertainty" here.  We are simply doing the above thought experiment and it
turns out we're maximizing :math:`E(-\log X)`.  In this manner, we might as
well give a name to :math:`-\log p`, which is the `Shannon information
<https://en.wikipedia.org/wiki/Information_content>`__ of a particular event.
This is nice because it doesn't require us to make any big leaps of assuming
that :math:`-\log p` has any meaning.

Physical Derivation 
===================

This derivation is from [3] which is not exactly a derivation of the concept of
entropy but the functional form.  It starts out with an observation in physical
systems involving a collection of equivalent elementary units where:

* Elementary units (e.g. particles) have some associated probability
  :math:`p_j` of taking on some numeric value :math:`j` (e.g. energy level),
  i.e., random variables.
* We observe some measurable quantity :math:`U` of the entire system (e.g. average temperature).
* The probability distribution of the elementary particles observed is the one
  maximizes the number of ways in which the particles can be arranged such that
  the system still measures :math:`U` (hint: this is the multiplicity :math:`W`
  from above, which is equivalent to maximum entropy).

We'll make this more precise, but first let's look at some examples.

* **Dice**: Given a die with :math:`j=1,2,3,...,m` faces, roll this die N
  times, compute the average value of the faces you see.  What you will find is
  that the maximum entropy principle predicts the probabilities of rolling each
  face of the die.  In general, this will be exponential or flat in the case of
  unbiased die.
* **Thermal system, canonical ensemble; temperature known**: Given N particles
  in a thermodynamic system, the numeric value of each particle is the energy
  state :math:`\varepsilon_j` of each particle.  Given a temperature T, which
  is equivalent to knowing the average energy, maximum entropy predicts 
  the Boltzmann distribution, :math:`p_j \propto \text{exp}[-\varepsilon_i/(kT)]`,
  which is what we observe.
* **Waiting Time Processes**: Consider you are watching cars pass by on a road
  and you measure the time between cars passing by as :math:`\tau_j`.  After
  observing :math:`N` cars, you measure the average waiting times between cars
  :math:`T/N = E(\tau)` where :math:`T` is the total waiting time.  What you
  will observe is that again maximum entropy predicts that the wait times will
  be exponentially distributed :math:`\text{exp}(-\lambda\tau_j)`.

In each of these situations maximum entropy is observed to be maximizing the
number of ways you can arrange the elementary units such that the given
constraint (:math:`U`) is satisfied.  In other words, we want to maximize the
quantity :math:`W` known as **multiplicity** which is the number of ways in
which the system can realize the observable :math:`U` from the elementary
units.

Defining Multiplicity
---------------------

Briefly repeating a variation of the previous section, if we
have :math:`N` elementary units, each of which can take on :math:`m` different
values, given a set of observations :math:`n_1, n_2, ... n_m` where
:math:`\sum_{i=1}^m n_i=N`, we can count the number of ways they can be arranged
as the multiplicity (same as Equation 5):

.. math::

   W(n_1, n_2, ... n_m) = \frac{N!}{n_1!\ldots n_m!} \tag{9}

Assuming that :math:`N` is large, we would expect :math:`\frac{n_i}{N} \approx p_i`,
the probability of each elementary unit taking on value :math:`i`.
Using an alternate form of 
`Sterling's approximation <https://en.wikipedia.org/wiki/Stirling%27s_approximation>`__
for large :math:`N` (we drop :math:`\sqrt{2\pi n}` factor since when we later take
logarithms it is negligible):

.. math::

   N! \approx \big( \frac{N}{e} \big)^N \tag{10}

Plugging this into Equation 9, we get:

.. math::

   W(n_1, n_2, ... n_m) &= \frac{N!}{n_1!\ldots n_m!} \\
    &\approx \frac{\big( \frac{N}{e} \big)^N}{
        (\big( \frac{n_1}{e} \big)^{n_1})
        (\big( \frac{n_2}{e} \big)^{n_2})
        \ldots
        (\big( \frac{n_3}{e} \big)^{n_3})} && \text{Sterling's approx.}\\
    &= (p_1^{-n_1}p_2^{-n_2}\ldots p_m^{-n_m}) && n_i = N p_i \\
    &= (p_1^{-p_1}p_2^{-p_2}\ldots p_m^{-p_m})^N \\
    &= W(p_1, p_2, \ldots, p_m) \\
   \tag{11}

Which you'll notice already resembles the exponentiated form of entropy we expect.

The definition of multiplicity in Equation 11 defines the number of ways the system
can realize particular values of :math:`n_1, n_2, \ldots, n_m`.  However, we
don't just want an arbitrary configuration, we want the one that satisfies our
observation :math:`U` (e.g. expected value).  That is, only count
configurations that satisfy the constraint :math:`U`.  We'll denote a
multiplicity that satisfies :math:`U` as :math:`W(p_1,\ldots, p_m, U)` 
or simply :math:`W(U)` when the context is clear.

Our goal now is to find the functional form of a new quantity we'll call
**entropy** :math:`S[W(p_1,\ldots, p_m, U)]`,
such that its extremum picks out the particular set of :math:`p_1,\ldots, p_m`
that maximize :math:`W(p_1,\ldots, p_m, U)`.
From here, you can already see that the logarithm of Equation 11 will probably
work out, but we'll show that this is actually the only choice that works.

Showing Entropy is Extensive
----------------------------

An extensive property :math:`P(\mathcal{R})` of a system :math:`\mathcal{R}` has these
conditions:


1. **Additivity**: If the system :math:`\mathcal{R}` can be divided into two subsystems :math:`\mathcal{R}_1`
   and :math:`\mathcal{R}_2` then:

   .. math::

      P(\mathcal{R}) = P(\mathcal{R}_1) + P(\mathcal{R}_2) \tag{12}


2. **Scalability**: If the size of the system :math:`\mathcal{R}` is scaled by a positive
   factor :math:`\alpha` then:
    
   .. math::

      P(\alpha \mathcal{R}) = \alpha P(\mathcal{R}) \tag{13}

We'll start by showing the first property since the second one follows from our end result. 

We wish to find entropy :math:`S(p_1, \ldots, p_m)` that is maximal where :math:`W` is
maximal that also satisfies the following conditions:

.. math::

   g(p_1, \ldots, p_m) &= \sum_{j=1}^m p_j = 1 && \text{probability constraint} \\
   h(p_1, \ldots, p_m) &= \sum_{j=1}^m x_j p_j = \frac{U}{N} && \text{observed measurement} \\
   \tag{14}

where :math:`x_j` is the :math:`j^{th}` value of the random variable for each
elementary unit.  Equation 14 just says that :math:`p_j` form a probability
distribution and that the multiplicity satisfies our observed measurement -- 
the average value of the observations (e.g. temperature).

Since we wish to find a maximum under constraints, we'll use 
`Lagrange multipliers <link://slug/lagrange-multipliers>`__.  Recall that we 
can set up the Lagrangian as:

.. math::

   \mathcal{L}(p_1, \ldots, p_m, \alpha, \lambda) = S(p_1, \ldots, p_m) - \alpha (g(p_1, \ldots, p_m) - 1) - \lambda (h(p_1, \ldots, p_m) - \frac{U}{N}) \tag{15}

where :math:`\alpha, \lambda` are our Lagrange multipliers for the constraints in Equation 14, 
which also include the constants on the RHS.  The extrema can be found by finding where
each of the partial derivatives equals to zero.  Taking the partial with respect to
:math:`p_j` and setting to zero gives us:

.. math::

   \frac{\partial\mathcal{L}(p_1, \ldots, p_m, \alpha, \lambda)}{\partial p_j} &= 0 \\
   \frac{\partial S(p_1, \ldots, p_m)}{\partial p_j} &= \frac{\partial}{\partial p_j} \big(\alpha (g(p_1, \ldots, p_m) - 1) + \lambda (h(p_1, \ldots, p_m) - \frac{U}{N}) \big)\\
   &= \frac{\partial}{\partial p_j} \big( \alpha (\sum_{j=1}^m p_j - 1) + \lambda (\sum_{j=1}^m x_j p_j - \frac{U}{N}) \big) \\
   \frac{\partial S(p_1, \ldots, p_m)}{\partial p_j}  &= \alpha + \lambda x_j \\
   \tag{16}

The `total differential <https://en.wikipedia.org/wiki/Total_derivative>`__ 
that gives the infinitesimal variation for :math:`S` can be written by plugging in Equation 16:

.. math::

   dS = \sum_{j=1}^m \frac{\partial S}{\partial p_j} dp_j = \sum_{j=1}^m (\alpha + \lambda x_j) dp_j \tag{17}

Now here comes the argument for why entropy is extensive: Let's arbitrarily partition
our system into two subsystems :math:`a` and :math:`b`.  Each subsystem will have
:math:`N_a` and :math:`N_b` elementary units (e.g. particles), each of which can
have multiplicity :math:`W_a(U_a)` and :math:`W_b(U_b)` respectively for given observations
:math:`U_a` and :math:`U_b`.  To make it even more general, the number of different values for
each subsystem can also be different with :math:`m_a` and :math:`m_b`
potentially being different.
Since it is a partition, we have :math:`N=N_a+N_B` and :math:`W(U) =
W_a(U_a)W_b(U_b)` where the second one follows from simply counting all the combined possibilities.

Similarly, each subsystem will have constraints that mirror Equation 14/16
(probability constraint and observed average value).  Thus, we can use Equation
17 to see that the total differential for each subsystem is:

.. math::

   dS_a = \sum_{j=1}^m (\alpha_a + \lambda_a x_{ja}) dp_{ja} \\
   dS_b = \sum_{j=1}^m (\alpha_b + \lambda_b x_{jb}) dp_{jb} \\
   \tag{18}
   
But since the two subsystems are a partition of the total system, we can write the total
differential for the entire system as a function of all the component parts
:math:`S(p_{1a},\ldots, p_{ma}, p_{1b}, \ldots, p_{mb})` with the four different constraints (two from each system):

.. math::

   dS &= \sum_{j=1}^{m_a} (\alpha_a + \lambda_a x_{ja}) dp_{ja} + \sum_{j=1}^{m_b} (\alpha_b + \lambda_b x_{jb}) dp_{jb} \\
   &= dS_a + dS_b \\
   \tag{19}

Notice that we did not make any assumptions about the form of entropy, the only
assumption we made is about the relation to a physical system.  Equation 19
shows (with some integration) that entropy is additive: 

.. math::

   S = S_a + S_b + C \tag{20}

where :math:`C` is a constant.  The scaling can be shown to be satisfied once
we find out that our functional form is a logarithm since increasing the number
of particles in a system by :math:`\alpha` exponentiates the multiplicity
:math:`W(U)^\alpha`.  Thus entropy is extensive.

Deriving the Functional Form of Entropy
---------------------------------------

Once we have shown entropy is additive, we can do some manipulation to show it must
have a logarithmic form.  First, let's simply the notation :math:`u := W_a(U_a),
v := W_b(U_b), r := W(U) = W_aW_b = uv`.  Rewriting Equation 20 with this notation:

.. math::

   S(r) = S_a(u) + S_b(v) + C \tag{21}

We can take the derivative of the left side with respect to :math:`v`:

.. math::

   \frac{dS}{dv} &= \frac{dS}{dr}\frac{\partial r}{\partial v} \\
                 &= \frac{dS}{dr}u \\
   \tag{22}

Now taking the derivative of the right hand side of Equation 21, we get:

.. math::

   \frac{d(S_a + S_b + C)}{dv} = \frac{dS_b}{dv} \tag{23}

Equating Equation 22/23:

.. math::

   \frac{dS}{dr}u = \frac{dS_b}{dv}  \tag{24}

Symmetrically if we take the derivatives with respect to :math:`u` in Equation
21, we also get:

.. math::

   \frac{dS}{dr}v = \frac{dS_a}{du}  \tag{25}

Equating Equation 24/25 using :math:`\frac{dS}{dr}`, we have:

.. math::

   u\frac{dS_a}{du} = v\frac{dS_b}{dv} = k \tag{26}

where :math:`k` is a constant.  The reason they are equal to a constant is the
left side is a function only of :math:`u`, while the right hand side is only a
function of :math:`v`, thus the only way two arbitrary functions of different
variables can be equal is if they are equal to the same constant.

Taking one side, we can solve the differential equation:

.. math::

   u\frac{dS_a}{du} &= k \\
   {dS_a} &= \frac{k}{u}{du} \\
   \int dS_a &= \int \frac{k}{u}{du} \\
   S_a &=  k\log{u} + C_a \\
   \tag{27}

where :math:`C_a` is the constant of integration.  You also get a similar
result for the other side.  Putting it together:

.. math::

   S(W) = S_a + S_b = k\log{W_a} + C_a + k\log{W_b} + C_b = k\log{W} + C' \tag{28}

We are free to choose the constant of integration such as :math:`S(1) = 0`,
which sets :math:`C'=0`.  Finally, plugging back the expression for :math:`W` 
from Equation 11 in:

.. math::

   S(W) &= k\log{(p_1^{-p_1}p_2^{-p_2}\ldots p_m^{-p_m})^N} \\
        &= -k'\sum_{j=1}^m p_j\log{p_j} && \text{define } k' = kN\\
        &= -\sum_{j=1}^m p_j\log{p_j} && \text{for } k = \frac{1}{N} \\
        \tag{29}

We can define :math:`k'` however we wish, so let's set it to :math:`k' = 1`,
thus we get the our expected expression for entropy.

Jaynes' Derivation
==================

Jaynes [1] has another derivation that is somewhat similar to the physical derivation
except he starts with desiderata of what we would like from an entropy measure.  Instead
of elementary particles, he shows using the rules of probability that an event can
be recursively broken down into "sub events" showing that entropy must be additive. 
From there, he is a bit more careful showing that entropy and the
multiplicity-equivalent variable would logarithmic if we assumed it to be continuous.
But since the inputs are integers (because they are multiplicities), you also
have to assume entropy is monotonically increasing with respect to the multiplicity.
In the end he shows that entropy is indeed logarithmic as expected.

I won't go into the gory details because it's quite involved and I think it's a
bit too technical to gain that much more intuition beyond the two derivations above.
Please do check out [1] though if you're interested, it's always a pleasure
reading Jaynes.

Conclusion
==========

Well I'm glad I got that post out of the way.  As soon as I read that appendix in [3],
I knew I had to write about the derivation.  Along the way I found Jaynes' derivation
in [1], which upon closer inspection also included the Wallis derivation.  As with every 
topic, you can go down an unlimited depth (and this one is a deep dive
on an already "elementary" topic).  For now, I'm satisfied with just explaining
two derivations, which give me a better appreciation for the beauty and
"surprise" of (maximum) entropy.  Stayed tuned for more (short to medium sized) posts!

References
==========

* [1] E. T. Jaynes, "`Probability Theory: The Logic of Science <https://doi.org/10.1017/CBO9780511790423>`__", Cambridge, 2006.
* [2] Wikipedia: `Principle of Maximum Entropy <https://en.wikipedia.org/wiki/Principle_of_maximum_entropy#The_Wallis_derivation>`__
* [3] Dill, K. A., & Bromberg, S. (2011). Molecular Dynamics (Appendix E). CRC Press.
