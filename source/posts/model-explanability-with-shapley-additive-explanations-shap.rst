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



|h2| Shapely Values and Cooperative Games |h2e|

Shapely Values and Cooperative Games 1.1.1
------------------------------------------

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
    data points we would look at the vector :math:`{\bf z'} = [1, 0]` 
    (where :math:`z \in {u, v}`) and use their :math:`h` mapping to find the
    original values:
    
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

        g({\bf z'}) = \phi_0 + \sum_{i=1}^M \phi_i z_i' \tag{13}

    where :math:`z' \in \{0,1\}^M`, :math:`M` is the number of simplified input
    features and :math:`\phi_i \in \mathbb{R}`.

This essentially captures our intuition on how to explain (in this case) a data
point: additive and independent.  Next, we'll look at some desirable properties
that we want to maintain in this mapping.

|h3| Desirable Properties and Shapely Values |h3e|

The first property we would want from our point-wise explanation model :math:`g(\cdot)`
is some guarantee of local accuracy:

    **Property 1 (Local Accuracy)**

    .. math::

        f(x) = g({\bf x'}) = \phi_0 + \sum_{i=1}^M \phi_i x_i' \tag{14}

    The explanation model :math:`g` matches the original model when :math:`x =
    h_x(x')`, where :math:`\phi_0 = h_x({\bf 0})` represents the model output
    with all simplified inputs toggled off.

All this property is saying is that if you pass in the original data point
with all features included (:math:`x`), your explanation model (:math:`g`) should
return the original value of your model, seems reasonable.

    **Property 2 (Missingness)**

    .. math::

        x_i' = 0 \implies \phi_i = 0 \tag{15}

    Missing input features in the original data point (:math:`x_i`) have no
    attributed impact.

This property almost seems unnecessary because all it is saying is that if your
original data point doesn't include the variable ("missing") then it shouldn't 
show any attributed impact in your explanation model.  The idea of a "missing"
input is still something we have to deal with because most models don't really
support missing feature columns.  We'll get to it in the next section.

    **Property 3 (Consistency)**: Let :math:`f_x(z') = f(h_x(z'))` and :math:`z' \backslash i` denote
    setting :math:`z_i'=0`.  For any two models :math:`f` and :math:`f'`, if
    
    .. math::

        f_x'(z')-f_x'(z' \backslash i) \geq f_x(z')-f_x(z' \backslash i) \tag{16}

    for all inputs :math:`z'\in {0,1}^M` then :math:`\phi_i(f', x) \geq \phi_i(f,x)`.

This is a more important property that essentially says: if we have two
(point-wise) models (:math:`f, f'`) and :math:`f'` consistently overweights a
certain feature :math:`i` in its prediction compared to :math:`f`, we would
want the coefficient of our explanation model for :math:`f'` to be bigger than
:math:`f` (i.e. :math:`\phi_i(f', x) \geq \phi_i(f,x)`).  It's a sensible
requirement that allows us to fairly compare different models using the same
explainability techniques.

These three properties lead us to this theorem:

    **Theorem 1** The only possible explanation model :math:`g` following a
    additive feature attribution method and satisfying Properties 1, 2, and 3
    are the shapely values from Equation 2:

    .. math::

        \phi_i(f,x) = \sum_{z'\subseteq x'} 
            \frac{|z'|!(M-|z'|-1)!}{M!}[f_x(z')-f_x(z' \backslash i)] \tag{17}
    
This is a bit of a surprising result since it's unique.  Interestingly, some
previous methods (e.g. Lime) don't actually satisfy all of these conditions
such as local accuracy and/or consistency.  That's why (at least theoretically)
Shapely values are such a nice solution to this feature attribution problem.

|h3| SHapely Additive exPlanations (SHAP) |h3e|

If it wasn't clear already, we're going to use Shapely values as our feature
attribution method called SHapely Additive exPlanations (SHAP).  From Theorem
1, we know that Shapely values provide the only unique solution to Properties
1-3 for a additive feature attribution model.  The big question is how do we
calculate the Shapely values (and what do they intuitively mean)?

Recall that Shapely rely on the value function, :math:`v(S)`, which determine
a mapping from a subset of features to an expected "payoff".  In the case of
Equation 17, our "payoff" is the model prediction :math:`f_x(z')` i.e.
the prediction of our model at point :math:`x` with subset of features
:math:`'z`.  Implicit in this definition is that we can evaluate our model with
just a subset of features, which most models do *not* support.  So how does SHAP 
deal with it? Expectations!  

To evaluate a model at point :math:`x` with a subset of features :math:`S`, it
starts out with the **expectation** of the function (recall, :math:`z'` is our binary
vector representing :math:`S` and :math:`h_x(\cdot)` is the mapping from the 
binary vector to the actual feature in data point :math:`x`):

.. math::

    f(h_x(z')) &= E[f(z)|z_S] \\
               &= E_{z_{\bar{S}}|z_S}[f(z)] \\
    \tag{18}

Of course, most models don't have an explicit probability density attached to them
so we have to approximate it using our dataset.  However, since our dataset is probably
not exhaustive, we probably won't be able to get a good estimate of
:math:`E_{z_{\bar{S}}|z_S}[\cdot]` because this means for every missing value
combination, we want to marginalize over the non-missing values.  For example,
if our data point is describing a customer with and we had one missing
dimension:

.. math::
    {\bf x}&={x_{sex}=M, x_{age=18}, x_{spending}=100, x_{recency}=2, x_{location}=City, \ldots} \\
    {\bf x_{\overline{sex}}}&={x_{sex}=N/A, x_{age=18}, x_{spending}=100, x_{recency}=2, x_{location}=City, \ldots} \\ 
    \tag{19}

then we would need to marginalize over every dimension except that one.  We
would likely not have enough data point to get a good estimate of
:math:`x_{\overline{sex}}` in this case (it also might be computationally
expensive).  Thus, we make some more simplifications, **independence** and
**linearity**:

.. math::

    f(h_x(z')) &= E[f(z)|z_S] \\
               &= E_{z_{\bar{S}}|z_S}[f(z)] \\
               &\approx E_{z_{\bar{S}}}[f(z)] && \text{independence} \\
               &\approx f(z_S, E[z_{\bar{S}}]) && \text{linear} \\
               \tag{20}

Independence allows us to treat each dimension separately and not care about the 
conditional aspect of trying to find a data point that "matches" :math:`x` (such as
in Equation 19}.
Linearity allows us to simply compute the expectation (:math:`E[z_{\bar{S}}]`)
over each dimension of :math:`x` separately and plug it into the model (as
opposed to evaluating the model each time over every dimension combination in
your dataset).

To summarize, we can calculate feature importance by:

1. Computing Shapely values as in Equation 17:

   .. math::

        \phi_i(f,x) = \sum_{z'\subseteq x'} 
            \frac{|z'|!(M-|z'|-1)!}{M!}[f_x(z')-f_x(z' \backslash i)]

2. To evaluate the model with "missing" values (:math:`f_x(z')`), we assume
   independence and linearity, which allows us to simply use the expectation
   (i.e. mean value) of each "missing" dimension and plug it into the model.

Now this leaves us with two additional questions: how do we interpret these
feature importances?  And how can we compute these values efficiently (without
running through all possible subsets of features).  We'll get to the first
question in the next subsection and the latter in the next section.

|h3| Interpreting SHAP Feature Importances |h3e|

.. figure:: /images/shap_values.png
  :width: 800px
  :alt: SHAP Values
  :align: center

  Figure 1: SHAP Values



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
