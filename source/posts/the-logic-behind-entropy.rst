.. title: The Logic of Entropy
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
Well I found a couple of good explanations about how to "derive" and thought
that I should share.

In this post, I'll be talking about the logic behind entropy.  Why
it is a reasonable thing to maximize and how we can interpret it from a few
different ways.  This post will be more math heavy, less code, but hopefully it
will give you more insight into this ubiquitous topic.


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
variable :math:`X` to a roll of the die.  If we are given samples of die rolls
then, with some appropriate prior, we can iteratively apply Bayes' theorem to
determine the probability :math:`P(X=i) = p_i` where :math:`i` is die face value.
This type of calculation is relatively straight forward, maybe tedious, but straight
forward.  But what if we only have a different type of information?

Now imagine we don't have samples of that die roll.  Instead we only know its mean
roll, that is :math:`E[X] = \mu_X`.  For example, if all the die faces had
equal probability we would get :

.. math::

   E[X] = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = 3.5 \tag{1}

So if :math:`\mu_X = 3.5` then we might reasonably guess that :math:`X` had a
uniform distribution.  

But what if :math:`\mu_X = 3`?  It could be that :math:`p_3=1.0`, but that
seems unlikely.  Maybe something like this is slightly more reasonable?

.. math:: 

   {\bf p} &= [0.2, 0.2, 0.2, 0.2, 0.2, 0] \\
   E[X] = 0.2(1 + 2 + 3 + 4 + 5) + 0(6) = 4 \tag{2}

But still that zero probability for :math:`p_6` seems kind of off.
One way to approach it is to apply a prior to each :math:`p_i` and
marginalizing over all of them to ensure it matches our information of
:math:`E[X] = \mu_X`.  But that only shifts the problem to finding the
right priors, not how to directly incorporate our information of :math:`E[X] = \mu_X`.

From this example, we have gleaned some important insights though.
Somehow concentrating all the mass together with :math:`p_3=1.0` is not exactly
what we want, and while Equation 2 might seem slightly more reasonable,
it also seems odd that we have :math:`p_6=0`. This gives us intuition that
we want to be conservative and "spread out" our probability, and at the same
time only assign something :math:`0` if it is truly ruled out as a probability.
This spreading allows us to be noncommittal about the distribution.  So
in some sense we want to maximize the spread given the constraint,
which already sounds like something `familiar <slug://maximum-entropy-distributions>`__ perhaps?
Let's keep going and find out.

Entropy and The Principle of Maximum Entropy
=============================================

Information theory `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`__
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
simple argument which through some calculation ends up with our tried and true
measure of entropy.  It's nice because it does not require defining information
as :math:`\log p` upfront and then making some argument that we want to
maximize it's expected value but gets there directly instead.

To start, let's assume the problem from the previous section where we wanted to
find the probability distribution that has :math:`m` discrete values given some
testable information about the distribution such as it's expected value or
variance (or more generally one of
its `moments <https://en.wikipedia.org/wiki/Moment_(mathematics)>`__).  Given
any candidate probability distribution represented by a vector 
:math:`{\bf p} = [p_1, \ldots, p_m]`, we can easily test to see if it satisfies
our constraint.  To this end, let's conduct the following random thought experiment:

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

   P({\bf p}) &= \frac{N!}{n_1!\ldots n_m!}q_1^{n_1}q_2^{n_2} \ldots q_m^{n_m} \\
   &= \frac{N!}{n_1!\ldots n_m!}m^{-N} 
   &&& \text{since } q_i = \frac{1}{m} \text{ and } \sum_{i=1}^m n_i = N\\
   \tag{4}

since :math:`m` is a constant in this problem, it suffices to maximize the first factor, which
we'll call the **multiplicity** of the outcome denoted by :math:`W`:

.. math::

   W = \frac{N!}{n_1!\ldots n_m!} \tag{5}

But of course, we can equivalently maximize a monotonically increasing function of :math:`W`,
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
   &= \lim_{N\to\infty} \frac{1}{N} \Big( N\log N - n - \mathcal{O}(\log N) 
       && \text{Sterling's approx.}\\
   &\hspace{4.5em} - \sum_{i=1}^m (N\cdot p_i)\log((N\cdot p_i)) - (N\cdot p_i) - \mathcal{O}(\log (N\cdot p_i)) \Big) \\
   &= \lim_{N\to\infty} \frac{1}{N} \Big( N\log N 
    - \sum_{i=1}^m (N\cdot p_i)\log((N\cdot p_i))  \Big) && \text{Drop lower order terms} \\
   &= \lim_{N\to\infty} \log N - \sum_{i=1}^m p_i\log((N\cdot p_i))   \\
   &= \lim_{N\to\infty} \log N - \log N \sum_{i=1}^m p_i - \sum_{i=1}^m p_i\log p_i   \\
   &= \lim_{N\to\infty} \log N - \log N - \sum_{i=1}^m p_i\log p_i   \\
   &= \lim_{N\to\infty} - \sum_{i=1}^m p_i\log p_i \\
   &= - \sum_{i=1}^m p_i\log p_i \\
   &= H({\bf p}) \\
   \tag{8}

Equation 8 shows that if we follow the logic of the above procedure, the "fair"
probability distribution is equivalent to maximizing the entropy.  Notice that
we did not mention "information", "surprise", or "uncertainty" here.
We are simply doing the above thought experiment and it turns out we're
maximizing :math:`E(-\log X)`.  In this manner, we might as well give
a name to :math:`-\log p`, which is the `Shannon information <https://en.wikipedia.org/wiki/Information_content>`__ of a particular event.  This is nice because it doesn't require
us to make any big leaps of assuming that :math:`-\log p` has any meaning.

Physical Derivation 
===================

This derivation is from [3] which is not exactly a derivation of the concept of
entropy but the functional form.  It starts out with an observation in physical
systems involving a collection of equivalent elementary units where:

* Elementary units (e.g. particles) can take on some associated probability
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
* **Thermal system, canonical ensemble; temperatur known**: Given N particles
  in a thermodynamic system, the numeric value of each particle is the energy
  state :math:`\varepsilon_j` of each particle.  Given a temperature T, which
  is equivalent to knowing the average energy, maximum entropy predicts 
  the Boltzmann distribution, :math:`p_j \propto \text{exp}[-\varepsilon_i/(kT)]`,
  which is what we observe.
* **Waiting Time Processes**: Consider you are watching cars pass by on a road
  and you measure the time between cars passing by as :math:`\tau_j`.  After
  observing :math:`N` cars, you measure the average waiting times between cars
  :math:`T/N = E(\tau)` where :math:`T` is the total waiting time..  What you
  will observe is that again maximum entropy predicts that the wait times will
  be exponentially distributed :math:`\text{exp}(-\lambda\tau_j)`.

In each of these situations maximum entropy is observed to be maximizing the
number of ways you can arrange the elementary units such that the given
constraint (:math:`U`) is satisfied.  In other words, we want to maximize the
quantity :math:`W` known as **multiplicity** which is the number of ways in
which the system can realize the observable :math:`U` from the elementary
units.

Briefly repeating the argument from the previous section, if we have :math:`N`
elementary units, each of which can take on :math:`m` different values, given a
set of observations :math:`n_1, n_2, ... n_m` where :math:`sum_{i=1}^m n_i=N`,
we can count the number of ways they can be arranged as the multiplicity (same
as Equation 5):

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

    

References
==========

* [1] E. T. Jaynes, "`Probability Theory: The Logic of Science <https://doi.org/10.1017/CBO9780511790423>`__", Cambridge, 2006.
* [2] Wikipedia: `Principle of Maximum Entropy <https://en.wikipedia.org/wiki/Principle_of_maximum_entropy#The_Wallis_derivation>`__
* [3] Dill, K. A., & Bromberg, S. (2011). Molecular Dynamics (Appendix E). CRC Press.
