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

.. contents::
    :local:


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

1. How important is each person to the overall goal of surviving longer 
   (i.e. what's their fair share of the winnings)?
2. What outcome can we reasonably expect?

This is essentially the problem of a cooperative game from game theory (this is
in contrast to the traditional non-cooperative games from game theory that most
people are familiar with).  Formally: 

    A **coalitional game** is a where there is a set :math:`P`
    (consisting of :math:`N` players) and a *value function* :math:`v`
    that maps subset of players to real numbers: :math:`v: 2^P \rightarrow
    \mathbb{R}` with :math:`v(\emptyset)=0` (empty set is zero).

The value function :math:`v(S)` describes the total expected payoff
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
For player :math:`i` and value function :math:`v`, the Shapely value
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

.. admonition:: Example 1: Glove Game

    Imagine we have a game where players have left or right handed gloves.
    The goal is to form pairs.  Imagine a three player game: 
    :math:`N=\{1,2,3\}` where Players 1 and 2 have left handed gloves and Player
    3 has a right handed glove.  The value function is scoring for every subset
    that has a L/R pair, and zero otherwise:

    .. math::
    
        v(S) = \begin{cases}
            1 & \text{if } S \in \{\{1,3\}, \{2,3\}, \{1,2,3\}\} \\
            0 & \text{otherwise}
        \end{cases}

    Using Equation 1, we need to find the marginal contribution of players.
    For Player 1, we have:

    .. math::

        v(\{1\}) - v(\emptyset) &= 0 - 0 &= 0 \\
        v(\{1,2\}) - v(\{2\}) &= 0 - 0 &= 0 \\
        v(\{1,3\}) - v(\{3\}) &= 1 - 0 &= 1 \\
        v(\{1,2,3\}) - v(\{2, 3\}) &= 1 - 1 &= 0 

    From Equation 1:

    .. math::

        \varphi_1(v) &= \frac{1}{N} \sum_{S \subseteq N \setminus \{i\}} 
            {N-1 \choose |S|}^{-1} (v(S\cup \{i\}) - v(S)) \\ 
         &= \frac{1}{3} \big[{2 \choose 0}^{-1} (v(\{1\}) - v(\emptyset))  +
             {2 \choose 1}^{-1} (v(\{1,2\}) - v(\{2\})) + \\
         &\hspace{2.3em}  {2 \choose 1}^{-1} (v(\{1,3\}) - v(\{3\})) +
            {2 \choose 2}^{-1} (v(\{1,2,3\}) - v(\{,,3\})) 
         \big]  \\ 
         &= \frac{1}{3}[1(0) + \frac{1}{2}(0) + \frac{1}{2}(1) + (1)(0)] \\
         &= \frac{1}{6} \\
         \tag{3}

    Based on symmetry (Property 2), we can conclude Player 1 and 2 are
    equivalent, so:
   
    .. math:: 

        \varphi_2(v) = \varphi_1(v) = \frac{1}{6} \\
        \tag{4}

    Further, due efficiency (Property 1), we can find the remaining Shapely
    value for Player 3:

    .. math::

        \varphi_3(v) = v(N) - \varphi_1(v) - \varphi_3(v) 
        = 1 - \frac{1}{6} - \frac{1}{6}
        = \frac{2}{3} \\
        \tag{4}

    As expected, since Player 3 has the only right handed glove, so their
    split of the profits should be 4 times bigger than the other players.


.. admonition:: Example 2: Business

    Consider an owner of the business, denoted by :math:`o`, who provides the
    initial investment in the business.  If there is no initial investment,
    then there is no business (zero return).  The business also has :math:`k`
    workers :math:`w_1, \ldots, w_k`, each of whom contribute :math:`p` to the
    overall profit.  Our set of players is:

    .. math::

        N = \{o, w_1, \ldots, w_k\} \tag{5}

    The value function for this game is:

    .. math::

        v(S) = \begin{cases}
            mp & \text{if } o \in S \\
            0 & \text{otherwise }
        \end{cases} \\
        \tag{6}

    where :math:`m` is the number of players in :math:`S \setminus o`.

    We can setup Equation 1 but breaking down the sum into symmetrical parts
    based on the size of :math:`S`:

    .. math::
         
        \varphi_{o}(v) &= \frac{1}{N} \sum_{S \subseteq N \setminus \{i\}} 
            {N-1 \choose |S|}^{-1} (v(S\cup \{i\}) - v(S)) \\ 
        &= \frac{1}{N} \big[
            \sum_{m=1}^k
            \sum_{S \subseteq N \setminus \{i\}, |S|=m} 
            {N-1 \choose |S|}^{-1} (v(S\cup \{i\}) - v(S)) 
            \big] \\ 
        &= \frac{1}{N} \big[
            \sum_{m=1}^k
            {N-1 \choose |S|}^{-1} {N-1 \choose |S|} mp
            \big] && \text{each of the inner summations is symmetric}\\ 
        &= \frac{1}{N} \sum_{m=1}^k mp \\ 
        &= \frac{1}{k+1} \frac{k(k+1)p}{2} && \text{N=k+1}\\ 
        &= \frac{kp}{2} \\ 
        \tag{7}

    By efficiency and symmetry, the rest of the k workers get the rest of the
    :math:`\frac{kp}{2}` profits (total profits is :math:`kp`), and thus each
    worker should get :math:`\frac{p}{2}` of the profits.  In other words, each
    worker is "contributing" half of their share of the profits they make to
    the business owner.  
    
    Is it fair though?  Shapely values seems to say so.  Whether this has any
    implications for our capitalistic society where investors/business owners
    get much more than half of the profits is left as an exercise for the
    reader :p

|h2| Feature Explainability as Shapely Values |h2e|


|h3| Additive Feature Attribution Models |h3e|

The first thing we need to cover is what does it mean for a model
to be explainable?  Take a simple linear regression for example:

.. math::

    y = \beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n  \tag{8}

This model probably follows our intuition of an explainable model:
each of the :math:`x_i` variables are *independent, additive*, and
we can clearly point to a coefficient (:math:`\beta_i`) saying how it
contributed to the overall result (:math:`y`).  But about more complex
models?  Even if we stick with linear models, as soon as we introduce any
interactions, it gets much more messy:

.. math::

    y = \beta_0 + \sum_{i=1}^n \beta_i x_i 
        + \sum_{i=1}^{n-1} \sum_{j>i}^{n} \beta_{i,j}x_i x_j
    \tag{9}

In this case, how much has :math:`x_i` contributed to the overall prediction?
I have no idea and the reason is that my intuition expects the variables to be
independent and additive, and if they're not, I'm not really sure how to "explain"
the model.

So far we've been trying to explain the model as a whole.  What if we did
something simpler?  What if instead of trying to explain the entire model,
we took on a simpler problem: explain a single data point.  Of course, we would
want some additional properties, namely, that we can interpret it like the
simple linear regression in Equation 8, that there is some guarantee of local
accuracy, and probably some guarantee that similar models would produce similar
explanations.  We'll get to all of that but first let's setup the problem and
go through some definitions.

What we are describing here are call **local methods** designed to explain a
single input :math:`\bf` on a prediction model :math:`f({\bf x})`.
When looking at a single data point, we don't really care about the level (i.e.
value) of the feature, just how much it contributes to the overall prediction.
To the end, let's define a binary vector of *simplified inputs* :math:`\bf x'`
(denoted by :math:`'`) that represents whether or not we want to include that
feature's contribution to the overall prediction (analogous to our cooperative
game above).  We also have a mapping function :math:`h_x({\bf x'}) = {\bf x}`
that translates this binary vector to the equivalent values
for the data point :math:`\bf x`.  Notice that this mapping function
:math:`h_x(\cdot)` is *specific* to data point :math:`x` -- you'll have one of
these functions for every data point.
It's a bit confusing to write out in words so let's take a look at an example,
which should clarify the idea.

.. admonition:: Example 3: Simplified Inputs

    Consider a simple linear interaction model with two real inputs 
    :math:`x_1, x_2`:

    .. math::

        y = 1 + 2 x_1 + 3 x_2 + 4 x_1 x_2  \tag{10}

    Let's look at two data points:

    .. math::

        {\bf u} &= (x_1, x_2) = (-1, -0.5), &&y = -0.5 \\
        {\bf v} &= (x_1, x_2) = (0.5, 1), &&y = 7 \\
        \tag{11}

    Let's suppose we wanted to look at the effect of :math:`x_1`, for these two
    data points we would look at the vector :math:`{\bf z'} = [1, 0]` and
    use their :math:`h` mapping to find the original values:
    
    .. math::

        h_{\bf u}({\bf z'}) = h_{\bf u}([1, 0]) = [-1, n/a] \\
        h_{\bf v}({\bf z'}) = h_{\bf v}([1, 0]) = [0.5, n/a] \\
        \tag{12}

    where we represent missing values with "n/a" (we'll get to this later).  As
    you can see, this formalism is just to allow us to speak about whether we
    should include a feature or not (:math:`\bf z'`) and their equivalent
    values (:math:`\bf u, v`).

Now that we have these definitions, our end goal is to essentially build
a new *explanation model*, :math:`g({\bf z'})` that ensures that 
:math:`g({\bf z'}) \approx f(h_{\bf x}({\bf z'}))` whenever 
:math:`{\bf z'} \approx {\bf x'}`.  In particular, we want this explanation
model to be simple like our linear regression in Equation 8.  Thus, let's define
this type of model:

    **Additive Feature Attribution Methods** have an explanation model that
    is a linear function of binary variables:

    .. math::

        g({\bf z'}) = \phi_0 + \sum_{i=1}^M \phi_i z_i'

    where :math:`z' \in \{0,1\}^M`, :math:`M` is the number of simplified input
    features and :math:`\phi_i \in \mathbb{R}`.


|h3| Desirable Properties and Shapely Values |h3e|

* Explain Property 1, 2, 3
* Show theorem 1, relate it back to our cooperative game


|h3| SHapely Additive exPlanations (SHAP) |h3e|

* Show Equation and simplifications
* Explain how to deal with missingness

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
