.. title: Hamiltonian Monte Carlo
.. slug: hamiltonian-monte-carlo
.. date: 2021-09-11 20:47:05 UTC-04:00
.. tags: 
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

|h2| Motivation and Background: Markov Chain Monte Carlo |h2e|


|h2| Hamiltonian Dynamics |h2e|

|h2| Hamiltonian Monte Carlo |h2e|

|h2| Experiments |h2e|

|h2| Conclusion |h2e|


|h2| Further Reading |h2e|

* Previous posts: `Markov Chain Monte Carlo Methods, Rejection Sampling and the Metropolis-Hastings Algorithm <link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__, 
* Wikipedia: 
* [1] Radford M. Neal, MCMC Using Hamiltonian dynamics, `arXiv:1206.1901 <https://arxiv.org/abs/1206.1901>`__, 2012.
