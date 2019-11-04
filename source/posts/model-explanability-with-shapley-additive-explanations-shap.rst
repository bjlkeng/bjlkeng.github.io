.. title: Model Explainability with SHapley Additive exPlanations (SHAP)
.. slug: model-explanability-with-shapley-additive-explanations-shap
.. date: 2019-11-01 07:24:22 UTC-04:00
.. tags: explainability, SHAP, game theory, mathjax
.. category: 
.. link: 
.. description: An *explanation* of SHapley Additive exPlanations (SHAP)
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

One of the big criticisms of modern machine learning is that it's essentially
blackbox -- data in, prediction out, that's it.  And in some sense, how could
it be any other way?  When you have highly non-linear with high degrees of
interactions, how can you possibly hope to have a simple understanding of what
the model is doing?  Well, turns out there is an interesting (and practical)
line of research along these lines.

This post will dive into the ideas of a popular technique published in the last
few years call *SHapely Additive exPlanations* (or SHAP).  It builds upon
previous work in this area by providing a unified framework to think
about explanation models as well as a new technique with this framework that
uses Shapely values.  I'll go over the math, the intuition, and some of the
more interesting applications. No need for an implementation because there is
already a nice little Python package! Confused yet?  Keep reading
and I'll *explain*.

.. TEASER_END

|h2| Shapely Values and Cooperative Games |h2e|

|h3| Cooperative Games |h3e|

Imagine you're on a reality TV show where you're stuck on a desert island with
two other people. The longer all of you stay on the island, the more money
you will all make.  You need to work together in order to maximize your return.
Your payoff will change significantly depending on whom you're with.
In this scenario, of course, you want to pick the two best people to bring with
you, but how to choose?  For example, you probably want to pick someone who has
extensive survivability experience, they would obviously contribute a lot to a
better outcome.  On the other hand, you probably don't want to pick someone
like me who spends the majority of their free time indoors in front a computer,
I might just be dead weight.  In essence, we want to answer two questions:

1. How important is each person to the overall goal of surviving longer?
2. What outcome can we reasonably expect?

This is essentially the problem of a cooperative game from game theory (this is
in contrast to the traditional non-cooperative games from game theory that most
people are familiar with).  Formally: 

    A **coalitional game** is a where there is a set :math:`P`
    (consisting of :math:`N` players) and a *characteristic function* :math:`v`
    that maps subset of players to real numbers: :math:`v: 2^P \rightarrow
    \mathbb{R}` with :math:`v(\emptyset)=0` (empty set is zero).

The characteristic function :math:`v(S)` describes the total expected payoff
for a subset of players :math:`S`.  For example, including a survivalist in
your desert island coalition will likely result in better outcome (payoff) than
including me.  This function answers the second question: what outcome can
we reasonably expect (given a subset of players).

So far we've just setup the problem though.  Now we want to answer the
question: what should be everyone's "fair" distribution of the total payoff?
In other words, the first question: how much does each person contribute to
the overall goal.  Clearly, the survivalist should get more more of the payout
than me, but how much more?  Is there only one "unique" solution?  The answer
lies in shapely values.

|h3| Shapely Values [1]_ |h3e|

**Shapely values** (named in honour of Lloyd Shapely), denoted by
:math:`\varphi_i(v)`, are a solution to the above coalition game.  
For player :math:`i` and characteristic function :math:`v`, the Shapely value
is defined as:

.. math::
    \varphi_i(v) &= \frac{1}{N} \sum_{S \subseteq N \setminus \{i\}} 
        {N-1 \choose |S|}^{-1} (v(S\cup \{i\}) - v(S)) \tag{1} \\ 
        &= \frac{1}{\text{number of players}} \sum_{\text{coalitions excluding }i}
            \frac{\text{marginal contribution of i to coalition}}{\text{number of coalitions excluding i of this size}} \\
        &= \sum_{S \subseteq N \setminus \{i\}} 
        \frac{|S|!(N-|S|-1)!}{N!} (v(S\cup \{i\}) - v(S)) \tag{2}

This definition is quite intuitive: average over your marginal contribution in
every possible situation. Equation 1 shows this intuition the best.
Equation 2 is a simplification that you might see more often, which is just
expanding the combination and simplifying.

Shapely vaules are also a very nice because they are the *only* solution with
these desirable properties:

1. **Efficiency**: The sum of Shapely values of all agents is equal to the total for the grand coalition:
    
   .. math::
       \sum_{i\in N} \varphi_i(v) = v(N)         

2. **Symmetry**: If :math:`i` and :math:`j` are two players who are equivalent in the sense that

   .. math::
       v(S \cup \{i\}) = v(S \cup \{j\})

   for every subset :math:`S` of :math:`N` which contain neither :math:`i`
   nor :math:`j`, then :math:`\varphi_i(v) = \varphi_j(v)`.

3. **Linearity**: Combining two coalition games :math:`v` and :math:`w` is linear
   for every :math:`i` in :math:`N`:

   .. math::

        \varphi_i(v+w) = \varphi_i(v) + \varphi_i(w) \\
        \varphi_i(av) = a\varphi_i(v)

4. **Null Player**: For a null player, defined as :math:`v(S\cup \{i\})=v(S)`
   for all coalitions :math:`S` not containing :math:`i`, then
   :math:`\varphi_i(v) = 0`.

All of these properties seem like pretty obvious things you would want to have:

* **Efficiency**: Of course, you want your distribution to players to
  actually sum up to the total reward.
* **Symmetry**: If two people contribute the same to the game, you want
  them to have the same payoff.
* **Linearity**: If a game is composed of two independent sub-games, you
  want the total game to be the sum of the two games.
* **Null Player**: If a player contributes nothing, then their share should
  be nothing.

Let's take a look at a couple examples to get a feel for it.

.. admonition:: Example 1: 

    TODO

.. admonition:: Example 2:

    TODO



|h2| Feature Explainability as Shapely Values |h2e|

|h2| Computing SHAP |h2e|

* Trees
* Linear Models
* Kernel Methods approximations

|h2| Applications |h2e|

|h2| Conclusion |h2e|


|h2| References |h2e|

* [1] "A Unified Approach to Interpreting Model Predictions", Scott M. Lundberg, Su-In Lee, `<http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions>`__
* [2] "Explainable AI for Trees: From Local Explanations to Global Understanding", Lundberg, et. al, `<https://arxiv.org/abs/1905.04610>`__
* [3] SHAP Python Package, `<https://github.com/slundberg/shap>`__
* Wikipedia: `<https://en.wikipedia.org/wiki/Shapley_value>`__



.. [1] This treatment of Shapely values is primarily a re-hash of the corresponding Wikipedia article.  It's actually written quite well, so you should go check it out!
