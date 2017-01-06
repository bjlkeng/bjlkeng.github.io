.. title: Maximum Entropy Distributions
.. slug: maximum-entropy-distributions
.. date: 2017-01-06 08:05:00 UTC-05:00
.. tags: probabilitiy, entropy, mathjax
.. category: 
.. link: 
.. description: A introduction to maximum entropy distributions.
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

This post will talk about a method to find the probability distribution that best
fits your given state of knowledge about some data.  Using the principle of maximum
entropy, and some testable information (e.g. the mean), you can find the
distribution that makes the fewest assumptions (the one with maximal
information entropy).  As you may have guessed, this is used often in Bayesian
inference to determine prior distributions and also (at least implicitly) in
natural language processing applications such as a maximum entropy (MaxEnt)
classifier (i.e. a multinomial logistic regression).  As usual, I'll go thorugh
some background, some intuition, and some math.  Hope you find this topic as
interesting as I do!

.. TEASER_END

|h2| Information Entropy |h2e|

There are a plethora of ways to intuitively understand information entropy,
I'll try to describe one that makes sense to me.  If it doesn't quite make
sense to you, I encourage you to find a few different sources until you can
piece together a picture that you can internalize.

Let's first clarify two important points about terminology.  First, information
entropy is a distinct idea from the physics concept of thermodynamic entropy.
There are parallels and connections have been made between the two ideas but
it's probably best to initially to treat them as separate things.  Second, the
"information" part refers to information theory, which deals with sending
messages (or symbols) over a channel.  One crucial point for our explanation is
that the "information" (or data source) is modelled as a probability
distribution.  So everything we talk about is with respect to a probabilistic
model of the data.

Now let's start from the basic idea of *information*.  Wikipedia has a good
article on `Shannon's rationale <https://en.wikipedia.org/wiki/Entropy_(information_theory)#Rationale>`_ 
when information, check it out for more details.  I'll simplify it a bit to
try to pick out the main points.

First, how much information is gained by observing (discrete) event :math:`i`
that has probability :math:`p_i`?  We would ideally like information (denoted by
:math:`I(p_i)`) to have certain properties:

1. :math:`I(p_i)` is anti-monotonic -- information increases when the probability of an
   event decreases, and vice versa.  If something almost always happens (e.g. the
   sun will rise tomorrow), then you really haven't gained much information; or
   if something very rarely happens (e.g. a gigantic earth quake), then more
   information is gained.
2. :math:`I(p_i=0)` is undefined -- for infintensimally small probability events,
   you have a infinitely large amount of information.
3. :math:`I(p_i)\geq 0` -- information is non-negative.
4. :math:`I(p_i=1)=0` -- sure things don't give you any information.
5. :math:`I(p_i, p_j)=I(p_i) + I(p_j)` -- for independent events :math:`i` and
   :math:`j`, information should be additive.  That is, getting the information
   (for independent events) together, or separately, should be the same.

Shannon discovered that the right choice to define information as:

.. math::

    I(p) := \log(1/p) = -\log(p) \tag{1}

The base of the logarithm isn't too important since it will just adjust the
value by a constant.  A usualy choice is base 2 which we'll usually call a
"bit", or base :math:`e`, which we'll call a "nat".

Now that we understand the idea of information for an event, we can define entropy.
For a given discrete probability distribution for random variable :math:`X`,
we define entropy of :math:`X` (denoted by :math:`H(X)`) as the expected value
of the information of :math:`X`:

.. math::

    H(X) := E[I(X)] &= \sum_{i=1}^n P(x_i)I(x_i) \\
    &= \sum_{i=1}^n p_i \log(1/p_i) \\
    &= -\sum_{i=1}^n p_i \log(p_i) \tag{2}

Eh voila!  The usual non-intuitive definition of entropy we all know and love.
Basically, entropy is the *average* amount of information we get for a given
probability distribution.

TODO: If we get have higher entropy, then things are easier to predict.  Less
entropy: uniform distribution.  Show a graph?  Explain some examples like this.



    

|h2| Principle of Maximum Entropy |h2e|



|h2| Further Reading |h2e|

* Wikipedia: `Maximum Entropy Probability Distribution <https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution>`_, `Principle of Maximum Entropy <https://en.wikipedia.org/wiki/Principle_of_maximum_entropy>`_, `Entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_
