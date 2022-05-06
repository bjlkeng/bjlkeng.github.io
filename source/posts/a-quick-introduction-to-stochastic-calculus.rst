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

Let's start with the formal definition of a `stochastic process <https://en.wikipedia.org/wiki/Stochastic_process#Stochastic_process>`__ from Wikipedia:

    A stochastic process is defined as a collection of random variables defined on a common `probability space  <https://en.wikipedia.org/wiki/Probability_space>`__
    :math:`(\Omega ,{\mathcal {F}},P)`, where :math:`\Omega` is a `sample space <https://en.wikipedia.org/wiki/Sample_space>`__,
    :math:`\mathcal {F}` is a :math:`\sigma`-`algebra <https://en.wikipedia.org/wiki/Sigma-algebra>`__, and :math:`P` is a
    `probability measure <https://en.wikipedia.org/wiki/Probability_measure>`__; and the random variables, indexed by some set
    :math:`T`, all take values in the same mathematical space :math:`S`,
    which must be `measurable <https://en.wikipedia.org/wiki/Measurable>`__
    with respect to some σ :math:`\sigma`-algebra` :math:`\Sigma`.

That's a mouthful!  Let's break this down and interpret the definition more intuitively
(skip this part if you're already familiar with the measure-theoretic probability definition).
First, we need to understand the definition of a probability space :math:`(\Omega ,{\mathcal {F}},P)`.
This is relatively intuitive.

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


* Discrete/continuous time, discrete/continuous variable

* Mathematical definition:
* https://en.wikipedia.org/wiki/Stochastic_process#Stochastic_process
* Explain a bit about probability
* Callout box of (simple) probability and measure-theoretic probability (Lebesque integrals)

* https://en.wikipedia.org/wiki/Stochastic_process#Examples
* Bernoulli
* Random Walk
* Weiner process

* Adapted Processes: https://en.wikipedia.org/wiki/Adapted_process
  * Itō integral, which only makes sense if the integrand is an adapted process. 

Weiner Processes
================

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
* Wikipedia: `Stochastic Processes <https://en.wikipedia.org/wiki/Stochastic_process#Stochastic_process>`__
* [1] Steven E. Shreve, "Stochastic Calculus for Finance II: Continuous Time Models", Springer, 2004.
