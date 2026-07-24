.. title: RL Math
.. slug: rl-math
.. date: 2026-07-22 15:30:46 UTC-04:00
.. tags: rl, reinforcement learning, math, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

It's been a long time coming, but I finally got around to gaining some
intuition on reinforcement learning.  At least in my experience, RL comes up a
lot less often in real world problem compared to supervised learning.  You
really have to have something that fits the shape of RL and that can't be
more easily solved (well enough) by something simpler like a `contextual
bandits <https://en.wikipedia.org/wiki/Multi-armed_bandit#Contextual_bandit>`__
or some more hacky method like throwing a classifier at it.  The other
motivation is that RL is driving a lot of the gains in LLM post-training
nowadays so I thought I should have a deeper understanding of it.


In the spirit of trying to keeping things shorter [#]_, this post is going to be a
concise primer on a few big ideas in reinforcement learning.  It uses a useful
frame that I got while watching Nathan Lambert's rlhf course [Lambert]_ and,
as usual, goes through the math to relate some of the key concepts to build
intuition.  For me at least, I dislike it when you are just presented with
disparate equations without understanding how you got to them.  Many ML
explanations either skip over the derivation, which builds intuition, or go way
overboard explaining the minutiae.  I'm hoping this post will strike a good
balance to give you a clear framework to understand some of the big concepts in
RL concisely with the usual caveat that this isn't my area of expertise and it
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

* Expectations
* Log derivative trick 
* Approx. an expectation


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
* :math:`\mathcal{P}` is the state transition probability function: :math:`\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)`
* :math:`\mathcal{R}` is the reward function: :math:`\mathcal{R}(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]`
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

    \pi(a|s) = P(A_t=a | S_t=s) \tag{1}

This is typically implemented as a neural network :math:`\pi_{\theta}` parameterized
by :math:`\theta` (you'll also see :math:`\pi^*`, where :math:`*` denotes the
theoretical optimal policy i.e., perfect decisions).  So our agent is basically
just a neural network that outputs probabilities on what action to take next.

To train the policy, we'll need data.  This usually comes in the form of a
**trajectory** (or **trace**) :math:`\tau` where you have observed an agent
(not necessarily your agent) operating in the environment.  Formally,
that shows up as a vector:

.. math::

   \tau = (S_0, A_0, R_0, S_1, A_1, R_1, \dots S_T) \tag{2}

This can be thought of as a "recording" of the session explored by the agent.
In the case that the traces are collected beforehand for training (via another
agent, an older version of our agent, randomly etc.), it's usually known as
**offline** training.  Otherwise if the agent is actively learning while
interacting with the environment it's called **online** training.

Finally, the expected **return** or cumulative discounted reward from time
:math:`t` to the end of the trace is given by:

.. math::

   G_t = \sum_{k=0}^{(T-t-1, \infty)} \gamma^k r_{t+k+1} \tag{3}

The cumulative reward :math:`G_t` at time :math:`t` basically adds up all the
rewards from now into the future discounted by a :math:`\gamma` factor.
When talking about the reward for a full trajectory :math:`\tau`, we often use
this notation:

.. math::

   R(\tau) = G_0 = \sum_{t=0}^{T-1} \gamma^t R_{t} \tag{4}

Similarly, for a policy :math:`\pi_\theta` sometimes we want to analyze the
probability of seeing an entire trace :math:`\tau`.  Since we have a 
memoryless Markov process, each decision by our agent is independent giving us:

.. math::

   p_\theta(\tau) = P(s_0)\Pi_{t=0}^{T-1} \pi_\theta(a_t | s_t) P(s_{t+1}|s_t,a_t) \tag{5}

where :math:`P(s_{t+1}|s_t,a_t)` is the probability of our environment moving
from :math:`s_t` to :math:`s_{t+1}` given action :math:`a_t`.  Notice the environmental
factors (:math:`P`) do not depend on :math:`\theta` so when we take the gradient, these will
drop out as we will see later.


V, Q, advantage, and Bellman equations
--------------------------------------


Policy Gradients
================


Value Methods
=============



References
==========

.. [Lambert] Nathan Lambert, "`Reinforcement Learning from Human Feedback <https://rlhfbook.com/course>`__", 2026.


Notes
=====

.. [#] I've also been debating the value of writing these posts given that everything here can be easily reproduced by an LLM (which in fact I had used to learn more about this subject).  But I ended deciding to write it for a couple of reasons: (1) It's helpful for me to digest what I've learned and take time to try to explain it again a la the `Feynman technique <https://en.wikipedia.org/wiki/Learning_by_teaching>`__, (2) I think it's still pedagogically useful to curate core ideas for posterity.  Besides, who else will feed the LLMs with raw unfiltered human text?  We can't the LLMs be trained on "junk data" from Twitter (aka X) and Reddit users.
