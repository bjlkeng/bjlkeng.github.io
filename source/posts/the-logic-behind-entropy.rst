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
with the probability of a quanta being assigned to an outcome being a constant `q_i = \frac{1}{m}`:

.. math::

   P({\bf p}) &= \frac{N!}{n_1!\ldots n_m!}q_1^{n_1}q_2^{n_2} \ldots q_m^{n_m} \\
   &= \frac{N!}{n_1!\ldots n_m!}m^{-N} 
   &&& \text{since } q_i = \frac{1}{m} \text{ and } \sum_{i=1}^m n_i = N\\
   \tag{3}

since :math:`m` is a constant in this problem, it suffices to maximize the first factor, which
we'll name multiplicity denoted by :math:`W`:

.. math::

   W = \frac{N!}{n_1!\ldots n_m!} \tag{4}


References
==========

* [1] E. T. Jaynes, "`Probability Theory: The Logic of Science <https://doi.org/10.1017/CBO9780511790423>`__", Cambridge, 2006.
* [2] Wikipedia: `Principle of Maximum Entropy <https://en.wikipedia.org/wiki/Principle_of_maximum_entropy#The_Wallis_derivation>`__

