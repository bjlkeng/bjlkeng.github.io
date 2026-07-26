.. title: A Reinforcement Learning Primer
.. slug: rl-math
.. date: 2026-07-22 15:30:46 UTC-04:00
.. tags: rl, reinforcement learning, math, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

It's been a long time coming, but I finally got around to really internalizing
a lot of the ideas from reinforcement learning.  In my experience, RL comes up a
lot less often in real world problem compared to supervised learning so I really 
never spent a lot of time digging into it simply because I didn't need to.
In cases where it might have fit, usually a simpler solution like a `contextual
bandits <https://en.wikipedia.org/wiki/Multi-armed_bandit#Contextual_bandit>`__
or some more hacky supervised learning method was good enough.  But now
that RL is used heavily in LLM post-training I thought I should have a deeper
(read mathematical) understanding of it.

In the spirit of trying to keeping things shorter [#]_, this post is going to be a
concise primer on a few big ideas in reinforcement learning.  It uses a useful
frame that I got while watching Nathan Lambert's RLHF course [Lambert]_ and,
as usual, builds up the math for some of the key concepts to gain a deeper intuition.
For me at least, I dislike it when you are just presented with disparate
equations without understanding how you got to them.  Many ML explanations
either skip over the derivation, which builds intuition, or go way overboard
explaining the minutiae.  I'm hoping this post will strike a good balance to
give you a clear framework to understand some of the big concepts in RL
concisely with the usual caveat that this isn't my area of expertise and it
won't be exhaustive.




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


Background
==========

Expectations, Monte Carlo Estimation, and Importance Sampling
-------------------------------------------------------------

We often want to find the expected value :math:`E_{x\sim P}[\Psi(x)]` of a function
:math:`\Psi(x)` of with respect to a distribution :math:`P`:

.. math::

    E_{x\sim P}[\Psi(x)] = \int f_P(x)\Psi(x) dx \tag{1}

where :math:`f_P(\cdot)` is the probability density function of :math:`P` 
(similarly we can use a sum and probability mass function instead when
:math:`P` is discrete).

Equation 1 is usually impossible to compute exactly because :math:`f_P(\cdot)` does not
usually have a closed form, for example when :math:`P` is our data distribution,
which we only ever observe data samples from.  In these cases, we can
approximate the expectation using IID samples :math:`x_i` from :math:`P` 
using `Monte Carlo estimation <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__ 
as such:

.. math::

    E_{x\sim X}[\Psi(x)] &= \int f_P(x)\Psi(x) dx \\
                      &\approx \frac{1}{N}\sum_{i=1}^N \Psi(x_i) \\
                      \\ \tag{2}

Notice that if we have *any* integral of this form, we can do the same
Monte Carlo approximation.  This shows up in 
`importance sampling <https://en.wikipedia.org/wiki/Importance_sampling>`__  
where instead of sampling from :math:`x\sim P`, we sample from a different
distribution (with the same support) :math:`x\sim Q` usually because it'll
either reduce variance or when sampling from :math:`P` is difficult (but can
still have the PDF of :math:`f_P(x)`).  

Of course, you can't just sample from :math:`Y` and do nothing else, you need
an weighting factor :math:`w(x)=\frac{f_P(x)}{f_Q(x)}` too.  Putting that
together with Equation 2, we get:

.. math::

    E_{x\sim P}[\Psi(x)] &= \int f_P(x) \Psi(x) dx \\
                      &= \int f_P(x) \frac{f_Q(x)}{f_Q(x)} \Psi(x)  dx \\
                      &= \int f_Q(x) [\frac{f_P(x)}{f_Q(x)} \Psi(x) ] dx \\
                      &= E_{x\sim Q}[\frac{f_P(x)}{f_Q(x)} \Psi(x) ] \\
                      &= E_{x\sim Q}[ w(x) \Psi(x)] \\
                      &\approx \frac{1}{N}\sum_{i=1}^N w(x) \Psi(x_i) \\
                      \tag{3}

This will come in handy later when we want to use data that wasn't generated 
directly by the current model being trained.

Log Derivative Trick
--------------------

The `log derivative trick <https://math.stackexchange.com/questions/2554749/whats-the-trick-in-log-derivative-trick>`__
is a common identity that is used in ML to rewrite our loss function to make
our computation simpler.  First take the log-likelihood function that we
typically get in ML :math:`\log \mathcal{L(\theta;x)} = \log p({\bf x | \theta})`, a
function of our model parameters :math:`\bf \theta` (recall :math:`\bf x` are our
training data thus constant).  We like to work in log-space because it converts
multiplications into additions, making the computation much simpler.

In neural networks and other areas, we often take the gradient of the
log-likelihood, also known as the 
`score function <https://en.wikipedia.org/wiki/Informant_(statistics)>`__ :math:`s(\theta;x)`,
although that name isn't used in ML contexts too often.  We can derive the
trick straight from the definition of the score:

.. math::

   \nabla \log p({\bf x | \theta}) &= \frac{\nabla p({\bf x | \theta})}{p({\bf x | \theta})} && \text{chain rule}\\
   \nabla p({\bf x | \theta}) &= p({\bf x | \theta}) \nabla \log p({\bf x | \theta}) \\
   \tag{4}

Notice the RHS of the second line now has the raw pdf/pmf :math:`p(\cdot)`,
multiplied by the log-likelihood.  When used inside an expectation, this allows
us to reparameterize the expectation and still use our good old log-likelihood.
This will make a bit more sense later when we use it.

Reinforcement Learning Basics
=============================

Reinforcement Learning and Markov Decision Processes
----------------------------------------------------

`Reinforcement learning (RL) <https://en.wikipedia.org/wiki/Reinforcement_learning>`__
is a branch of machine learning whose goal is to learn an intelligent agent 
(not to be confused with the typical `generative AI agent <https://en.wikipedia.org/wiki/AI_agent>`__)
that can take actions in a dynamic environment in order to maximize a reward
signal.  The prototypical example of an (RL) agent is a game-playing AI that
selects the next action (e.g. left, right, shoot, etc.) in a computer game where
the reward is your score in the game, which you're trying to maximize.

As usual, we'll formalize this concept with a **Markov Decision Process (MDP)**
as a 5-tuple :math:`\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)`, where:

* :math:`\mathcal{S}` is a finite set of states.
* :math:`\mathcal{A}` is a finite set of actions.
* :math:`\mathcal{P}` is the state transition probability function: :math:`\mathcal{P}(s' \mid s, a) = P(S_{t+1} = s' \mid S_t = s, A_t = a)`
* :math:`\mathcal{R}` is the reward function: :math:`\mathcal{R}(s, a) = E[R_{t+1} \mid S_t = s, A_t = a]`
* :math:`\gamma \in [0, 1]` is the discount factor.

Even though it seems like a lot of math, it's actually pretty intuitive. Let's
break it down.
First, it's **"Markov"** because this describes the transition probability
function :math:`\mathcal{P}` through states :math:`\mathcal{S}` -- a `Markov chain
<https://en.wikipedia.org/wiki/Markov_chain>`__.  It's basically a "finite
state machine with probabilistic transitions".  See my `previous post
<link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__
for an example of Markov Chains.  In our game example, this defines the game
state and how it evolves (potentially non-deterministically) over time (based
on your input player actions).

Next, it's a **Decision Process** because the RL **agent** is concerned with not just
traversing probabilistically through states (Markov chain) but also taking
successive actions :math:`\mathcal{A}` in those states to achieve goal (reward
:math:`\mathcal{R}`).  It takes these actions in the context of an
**environment** defined by the state transitions and rewards.  
The **discount factor** :math:`\gamma` defines how much we value rewards we collect
later vs. now.  In our game example, this is agent represents the player whose
purpose is to maximize the "score" and the environment is the game.  

The following table summarizes this game analogy more concisely.  We'll get more
into the mathematical details in the next subsection.

.. raw:: html

   <style>
   table.rl-table thead th {
       font-size: 1.4rem;
       font-weight: 700;
       padding: 0.0  0.4rem;
   }
   </style>


.. list-table:: Table 1: Reinforcement Learning Terminology vs. Game Analogy
   :widths: 35 65
   :header-rows: 1
   :class: rl-table

   * * Reinforcement Learning
     * Game Analogy
   * * **Agent**
     * The player controlled by the AI.
   * * **Environment**
     * The game itself, including its rules, physics, enemies, and scoring system.
   * * **State** (:math:`s \in \mathcal{S}`)
     * The current game situation, such as the player's position, health, inventory, and nearby enemies.
   * * **Action** (:math:`a \in \mathcal{A}`)
     * A move available to the player, such as moving left, jumping, shooting, or using an item.
   * * | **State-transition function**  
       | ( :math:`\mathcal{P}(s' \mid s,a)`)
     * The rules governing how the game changes after the player takes an action. The result may be deterministic or involve randomness.
   * * **Reward function** (:math:`\mathcal{R}(s,a)`)
     * The points or other feedback received after an action, such as earning points for defeating an enemy or losing points for taking damage.
   * * **Discount factor** (:math:`\gamma`)
     * How much the player values points earned later compared with points earned immediately.
  

Policies, Trajectories and Returns
----------------------------------

Now with the formal definition of an MDP out of the way, let's look
at how we actually create an RL agent.  The big idea is that all the
agent does is iteratively *select the next action* :math:`a_t` at time :math:`t`.
As it selects the next action, the environment (a blackbox) updates the state
:math:`s_{t+1}` we observe and tells us our reward :math:`r_t` for taking the
action at that state (:math:`s_t, a_t`).
Thus, an RL agent is just something that is able to to pick the next action
given the current state.  

This is usually implemented as a mapping from state to a probability
distribution over actions called a **policy**:

.. math::

    \pi(a|s) = P(A_t=a | S_t=s) \tag{5}

This is typically implemented as a neural network :math:`\pi_{\theta}` parameterized
by :math:`\theta` (you'll also see :math:`\pi^*`, where :math:`*` denotes the
theoretical optimal policy i.e., perfect decisions).  So our agent is basically
just a neural network that outputs probabilities on what action to take next.

To train the policy, we'll need data.  This usually comes in the form of a
**trajectory** (or **trace**) :math:`\tau` where you have observed an agent
(not necessarily your agent) operating in the environment.  Formally,
that shows up as a vector:

.. math::

   \tau = (S_0, A_0, R_0, S_1, A_1, R_1, \dots S_T) \tag{6}

This can be thought of as a "recording" of the session explored by the agent.
In the case that the traces are collected beforehand for training (via another
agent, an older version of our agent, randomly etc.), it's usually known as
**offline** training.  Otherwise if the agent is actively learning while
interacting with the environment it's called **online** training.

Finally, the expected **return** or cumulative discounted reward from time
:math:`t` to the end of the trajectory is given by:

.. math::

   G_t = \sum_{k=0}^{(T-t-1, \infty)} \gamma^k r_{t+k+1} \tag{7}

The return :math:`G_t` at time :math:`t` basically adds up all the
rewards from now into the future discounted by appropriately by powers of
:math:`\gamma`.
When talking about the return for a full trajectory :math:`\tau`, we often use
this notation:

.. math::

   R(\tau) = G_0 = \sum_{t=0}^{T-1} \gamma^t r_{t} \tag{8}

Similarly, for a policy :math:`\pi_\theta` sometimes we want to analyze the
probability of seeing an entire trajectory :math:`\tau`.  Since we have a 
memoryless Markov process, we can nicely factor our joint trajectory distribution
like so:

.. math::
   
   p_\theta(\tau) = P(s_0)\Pi_{t=0}^{T-1} \pi_\theta(a_t | s_t) P(s_{t+1}|s_t,a_t) \tag{9}

where :math:`P(s_{t+1}|s_t,a_t)` is the probability of our environment moving
from :math:`s_t` to :math:`s_{t+1}` given action :math:`a_t`.  Notice the environmental
factors (:math:`P`) do not depend on :math:`\theta` so when we take the gradient, these will
drop out as we will see later.

Using Equation 8 and 9, our ultimate goal is to maximize the total expected
return of our policy:

.. math::

   E_{\tau \sim p_\theta}[R(\tau)] = \int R(\tau) p_\theta(\tau) d\tau \tag{10}

We'll see that we rarely use Equation 10 to learn our policy because its Monte
Carlo estimator's variance is too high to use directly.


V, Q, Bellman, and Advantage Equations
--------------------------------------

We'll also need to define a few more quantities that will be useful in our
practical implementations of RL.  First up is the **value function** that
estimates the expected future return from time :math:`t`:

.. math::

   V^\pi(s) = E_\pi [ G_t | S_t = s ] \tag{11}

During training, we'll need an estimate of the return from current and next
states to be able to robustly generate a stable learning signal efficiently.
Notice that the value function is with respect to a policy since changing the
policy will change the future return.  Similarly, we rarely have the exact
value function for a policy, so similarly, we often will learn the value
function parameterized by :math:`\phi` and denote it as :math:`V_\phi(s)`.

Next, we have the **Q-function** or action-value function is similar to the
value function except it estimates the return in state :math:`s` from taking
action :math:`a` given by:

.. math::

   Q^\pi(s,a) &= E_\pi [ G_t | S_t = s, A_t = a ]  \\
              &= E_\pi[r_t + \gamma V^\pi(s') | S_t = s, A_t = a] \\
   \tag{12}

where math:`s'` is the next state resulting from taking action :math:`a` in
state :math:`s`.  The Q function can be written in terms of the value function
since its return is just the current action/state's return plus the future
returns from the next state.  We need the expectation because the environment
can be stochastic so we need to average over the future potential
rewards/trajectories.

Similar to the value function, the Q-function can also be learned and used to
help understand how a policy is doing, although it's less commonly used than
the value function.  More commonly, the Q-function can be used directly to
derive a policy since if you have a good estimate of the q-function, you can
just take the most profitable action at any given state.  We'll get more into
this in later sections.

Moving on, MDPs have `optimal substructure
<https://en.wikipedia.org/wiki/Optimal_substructure>`__ because of its Markov
property, which means that you can construct an optimal solution to a bigger
problem given optimal solutions to its subproblems.  All dynamic programming
problems have optimal substructure.  This means for our sequential decision
making RL problem, we can use **Bellman's equation**, shown here for the
optimal (:math:`*`) and policy (:math:`\pi`) versions of the Q-function:

.. math::

   Q^*(s, a) &= E[r_t + \gamma \max_{a'} Q^*(s', a') | S_t = s, A_t = a ]
   \\ 
   Q^\pi(s, a) &= E_\pi[r_t + \gamma \max_{a'} Q^\pi(s', a') | S_t=s, A_t=a]
   \tag{13}

Similar to dynamic programming, we can iteratively apply the Bellman equations
to get convergence.  The optimal version guarantees convergence to optimality,
but the policy version has a weaker guarantee to converge to the true value of
the policy.  We'll use these to derive an update rule for our learning
algorithms later.

Lastly, we have the **advantage function** :math:`A^\pi` [#]_, which defines how much
better a specific action is over the average in a current state.  Using the
above equations we can write it out and expand for given policy :math:`\pi`:

.. math::

   A^\pi(s,a) &= Q^\pi(s, a) - V^\pi(s) \\
    &= E_\pi[r_t + \gamma V^\pi(s') | S_t = s, A_t = a] - V^\pi(s) && \text{Equation 12} \\
   \tag{14}

We'll use the second version along with a separate value function network in
later sections to help make a lower variance estimator compared to Equation 10.


Policy Gradients
================

Policy gradients [#]_ are a type of reinforcement learning that directly aims to
learn a policy :math:`\pi_\theta(a|s)`, usually a neural network parameterized
by :math:`\theta`.  It's input is the observed current state of the environment
:math:`s` and its output is a probability distribution over possible actions
:math:`a`.  This forms the agent which sequentially makes decisions to maximize
reward in a given environment.

As with more neural network based approaches, we update the :math:`\pi_\theta`
with a scaled version of its gradient (hence the name policy gradients) based
on a loss derived from total expected return of our policy in Equation 10.
The general form of the update is:

.. math::

    \Delta \theta \propto \Psi_t \nabla_\theta \log \pi_\theta(a_t|s_t) \tag{15}

where :math:`\Psi_t` relates to how good :math:`a_t` was as an action
(implicitly being related to the return).  It's non-obvious why Equation 15 is
our update so we'll spend time deriving it in the first sub-section.  The rest
of the subsections will focus on the :math:`\Psi` and other "tricks" to make
Equation 15 more viable as a policy gradient.

Deriving the Policy Gradient
----------------------------

The first step in deriving our policy gradient is to define our loss function.
We start at our total expected return in Equation 10  as our loss function:

.. math::

   J(\theta) = E_{\tau \sim p_\theta}[R(\tau)] = \int R(\tau) p_\theta(\tau) d\tau \tag{16}

Now taking the gradient:

.. math::

   \nabla_\theta J(\theta) &= \nabla_\theta \int R(\tau) p_\theta(\tau) d\tau  \\
   &= \int R(\tau) \nabla_\theta p_\theta(\tau) d\tau  \\
   \tag{17}

At this stage, we can't derive a Monte Carlo estimator like in Equation 2 because
:math:`\nabla_\theta p_\theta(\tau)` is not a probability distribution after we apply the
gradient operation.  But we can apply our log derivative trick from Equation 4 to get:

.. math::

   \nabla_\theta J(\theta) &= \int R(\tau) \nabla_\theta p_\theta(\tau) d\tau   \\
   &=  \int R(\tau) p_\theta (\tau) \nabla_\theta \log p_\theta(\tau) d\tau && \text{Equation 4}   \\
   \tag{18}

And that gives us something that we can use Equation 4 on to get a Monte Carlo estimator!
However, notice that our gradient update in Equation 15 uses actions :math:`a` and states :math:`s`,
not full trajectories :math:`\tau`.  To solve that, we'll need to use Equation 9 which writes
the probability of the full trajectory in terms of the state-action transitions.  Starting
from the gradient of the log policy probability from Equation 18:

.. math::

   \nabla_\theta \log p_\theta(\tau)
   &= \nabla_\theta \log \big[ P(s_0)\Pi_{t=0}^{T-1} \pi_\theta(a_t | s_t) P(s_{t+1}|s_t,a_t) \big] && \text{Equation 9} \\
   &= \nabla_\theta \big( \log P(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t | s_t) + \log P(s_{t+1}|s_t,a_t) \big) \\
   &= \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) && \text{only }\pi_\theta \text{ is a function of } \theta \\
   \tag{19}

Thus, we get the same expression that we saw the generic policy gradient update in Equation 15.

As a last step, we need to finally use a Monte Carlo estimator to turn our loss integral into
something we can sample. Starting from Equation 18:

.. math::

   \nabla_\theta J(\theta) &= \int R(\tau) p_\theta (\tau) \nabla_\theta \log p_\theta(\tau) d\tau  \\
   &= \int R(\tau) p_\theta (\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) d\tau \\
   &\approx \frac{1}{N} \sum_{i=1}^N R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) && \text{Equation 2} \\
   &\approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} R(\tau) \nabla_\theta \log \pi_\theta(a_t | s_t) \\
    \Delta \theta &\propto R(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t) \\
   \tag{20}

So with the Monte Carlo estimator we can iteratively use the inner expression
to update our model's parameters as long as we have enough samples to converge
to the original expectation.  That last condition is load bearing though
because as it stands the estimator for Equation 20 has very high variance.
That's why we'll make adjustments to Equation 20 (like replacing :math:`R(\tau)`)
to lower the variance without (most of the time) affecting the bias.


Value Methods
=============



References
==========

.. [Lambert] Nathan Lambert, "`Reinforcement Learning from Human Feedback <https://rlhfbook.com/course>`__", 2026.


Notes
=====

.. [#] I've also been debating the value of writing these posts given that everything here can be easily reproduced by an LLM (which in fact I had used to learn more about this subject).  But I ended deciding to write it for a couple of reasons: (1) It's helpful for me to digest what I've learned and take time to try to explain it again a la the `Feynman technique <https://en.wikipedia.org/wiki/Learning_by_teaching>`__, (2) I think it's still pedagogically useful to curate core ideas for posterity.  Besides, who else will feed the LLMs with raw unfiltered human text?  We can't the LLMs be trained on "junk data" from Twitter (aka X) and Reddit users.
.. [#] I know, I know, action :math:`A_t` and advantage :math:`A^\pi` both use the same letter but that's the convention, and usually it's easy to understand the difference from context.
.. [#] Much of this section was derived from Nathan Lambert's RL lecture from his RLHF course [Lambert]_.  After watching his lecture, it inspired me to dig deeper into the math to understand it all in detail.  As I usually do, I expanded on it a bit to present it in a way that answers questions that I had along the way.
