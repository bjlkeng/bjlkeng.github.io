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
