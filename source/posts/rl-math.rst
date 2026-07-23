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

Reinforcement Learning and Markov Decision Processes
----------------------------------------------------

`Reinforcement learning (RL) <https://en.wikipedia.org/wiki/Reinforcement_learning>`__
is a branch of machine learning whose goal is to learn an intelligent agent 
(not to be confused with the typical `generative AI agent <https://en.wikipedia.org/wiki/AI_agent>`__)
that can take actions in a dynamic environment in order to maximize a reward
signal.  The prototypical example of an (RL) agent is a game-playing AI that
selects the next action (e.g. left, right, shoot, etc.) in a computer game where
the reward is your score in the game, which you're trying to maximize.

This all can be modelled using a **Markov Decision Process (MDP)**.
Formally, it is a 5-tuple :math:`\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)`, where:

* :math:`\mathcal{S}` is a finite set of states.
* :math:`\mathcal{A}` is a finite set of actions.
* :math:`\mathcal{P}` is the state transition probability function: :math:`\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)`
* :math:`\mathcal{R}` is the reward function: :math:`\mathcal{R}(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]`
* :math:`\gamma \in [0, 1]` is the discount factor.

Even though it seems like a lot of math, it's actually pretty intuitive. Let's
break it down.
First, it's **"Markov"** because this describes the transition probability
function :math:`\mathcal{P}` through states `\mathcal{S}` -- a `Markov chain
<https://en.wikipedia.org/wiki/Markov_chain>`__.  It's basically a "finite
state machine with probabilistic transitions".  See my `previous post
<link://slug/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm>`__
for an example of Markov Chains.  In our game example, this defines the game
state and how it evolves (non-deterministically) over time (based on your input
player actions).

Next, it's a **Decision Process** because the RL **agent** is concerned with not just
traversing probabilistically through states (Markov chain) but also taking
successive actions :math:`\mathcal{A}` in those states to achieve goal (reward
:math:`\mathcal{R}`).  It takes these actions in the context of an
**environment** defined by the state transitions and rewards.
In our game example, this is agent represents the player whose purpose is to
maximize the "score" and the environment is the actual game.

A **trajectory** (or **trace**) :math:`\tau` of this process is the sequence of
states, actions, and rewards experienced by the agent over time:

.. math::

   \tau = S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots' \tag{2}

This is basically a "recording" of the session played by our AI in the game.
It forms the nuts and bolts of our training data and can be generated either
(a) offline (via another agent, an older version of our agent, randomly etc.), 
or (b) online where our agent actively learning while it is interacting with
the environment.

Finally, the aim of a RL agent is to find an optimal **policy** :math:`\pi^*`
that maximizes the expected **cumulative discounted reward**:

.. math::

   G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \tag{1}

We use :math:`\pi_\theta` usually to denote a policy, which is usually a neural
network that has parameters :math:`\theta` (:math:`*` when it's the theoretical
optimal policy -- it always makes perfect decisions).  The cumulative reward
:math:`G_t` at time :math:`t` basically adds up all the rewards from now into
the future discounted by :math:`\gamma`.

When viewed from the lens of a simple game, the terms have a clear intuitive
mapping.  Things get a lot more complicated with real world problems.  For
example, near-continuous action spaces (e.g. setting a price), highly noisy
non-stationary environments (e.g. stock market), high-dimensional state spaces
(e.g. raw camera pixels), sparse rewards (e.g. only available at the end of a
long trajectory), and expensive data collection (e.g. robotics).  I won't be
covering any real solutions to these challenges since I'm mostly explaining the
basics, but know that RL is challenging to apply to real world problems.


References
==========

.. [Lambert] Nathan Lambert, "`Reinforcement Learning from Human Feedback <https://rlhfbook.com/course>`__", 2026.


Notes
=====

.. [#] I've also been debating the value of writing these posts given that everything here can be easily reproduced by an LLM (which in fact I had used to learn more about this subject).  But I ended deciding to write it for a couple of reasons: (1) It's helpful for me to digest what I've learned and take time to try to explain it again a la the `Feynman technique <https://en.wikipedia.org/wiki/Learning_by_teaching>`__, (2) I think it's still pedagogically useful to curate core ideas for posterity.  Besides, who else will feed the LLMs with raw unfiltered human text?  We can't the LLMs be trained on "junk data" from Twitter (aka X) and Reddit users.
