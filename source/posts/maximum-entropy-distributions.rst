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

|h2| Information Entropy and Differential Entropy |h2e|

There are plenty of ways to intuitively understand information entropy,
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

Now let's start from the basic idea of `*information* <https://en.wikipedia.org/wiki/Self-information#Definition>`.  Wikipedia has a good
article on `Shannon's rationale <https://en.wikipedia.org/wiki/Entropy_(information_theory)#Rationale>`_ 
when information, check it out for more details.  I'll simplify it a bit to
try to pick out the main points.

First, *information* was originally defined in the context of sending a message
between a transmitter and receiver over a (potentially noisy) channel.
Think about a situation where you are shouting messages to your friend
across a large field.  You are the transmitter, your friend the receiver, and
the channel is this large field.  We can model what your friend is hearing
using probability.  

For simplicity, let's say you are only shouting (or transmitting) letters of
the alphabet (A-Z).  We'll also assume that the message always transmits
clearly (if not this will affect your probability distribution by adding noise).
Let's take a look at a couple examples to get a feel for how information works:

1. Suppose you and your friend agree that you will always shout "A" ahead of
   time (or a priori).  So when you actually do start shouting, how much
   information is being transmitted?  None, because your friend knows exactly
   what you are saying.  This is akin to modelling the probability of receiving
   "A" as 1, and all other letters as 0.
2. Suppose you and your friend agree, a priori, that you will being shouting
   letters in order from some English text.  Which letter do you think would
   have more information, "E" or "Z"?  Since we know "E" is the most common letter
   in the English language, we can usually guess when the next character is an
   "E".  So we'll be less surprised when it happens, thus it has a relatively
   low amount of information that is being transmitted.  Conversly, "Z" is an
   uncommon letter.  So we would probably not guess that it's coming next and be
   surprised when it does, thus "Z" conveys more information than "E" in this
   situation.  This is akin to modelling a probability distribution over the
   alphabet with probabilities proportional to the relative frequencies of letters
   occurring in the English language.

When we have the probability of an event, we can generalize this idea to define
information as a number like so:

.. math::

    I(p) := \log(1/p) = -\log(p) \tag{1}

The base of the logarithm isn't too important since it will just adjust the
value by a constant.  A usualy choice is base 2 which we'll usually call a
"bit", or base :math:`e`, which we'll call a "nat".

.. admonition:: Properties of Information

    The definition of information came about based on certain definitions of
    how we expect information to behave:
    
    1. :math:`I(p_i)` is anti-monotonic - information increases when the probability of an
       event decreases, and vice versa.  If something almost always happens (e.g. the
       sun will rise tomorrow), then you really haven't gained much information; or
       if something very rarely happens (e.g. a gigantic earth quake), then more
       information is gained.
    2. :math:`I(p_i=0)` is undefined - for infintensimally small probability events,
       you have a infinitely large amount of information.
    3. :math:`I(p_i)\geq 0` - information is non-negative.
    4. :math:`I(p_i=1)=0` - sure things don't give you any information.
    5. :math:`I(p_i, p_j)=I(p_i) + I(p_j)` - for independent events :math:`i` and
       :math:`j`, information should be additive.  That is, getting the information
       (for independent events) together, or separately, should be the same.

Now that we have an idea about the information of a single event, we can define
entropy in the context of a probability distribution (over a set of events).
For a given discrete probability distribution for random variable :math:`X`,
we define entropy of :math:`X` (denoted by :math:`H(X)`) as the expected value
of the information of :math:`X`:

.. math::

    H(X) := E[I(X)] &= \sum_{i=1}^n P(x_i)I(x_i) \\
    &= \sum_{i=1}^n p_i \log(1/p_i) \\
    &= -\sum_{i=1}^n p_i \log(p_i) \tag{2}

Eh voila!  The usual (non-intuitive) definition of entropy we all know and
love.  Basically, entropy is the *average* amount of information an event in a
given probability distribution.
Going back to our example above, when transmitting only "A"s, the average
information transmitted is 0, so the entropy is naturally 0.
When transmitting English text, the entropy will be the average entropy
using `letter frequencies <https://en.wikipedia.org/wiki/Letter_frequency#Relative_frequencies_of_letters_in_the_English_language>`_ [1]_.


.. admonition:: Example 1: Entropy of a fair coin.

    For a random variable X corresponding to the toss of a fair coin we have,
    :math:`P(X=H)=p` and :math:`P(X=T)=1-p` with :math:`p=0.5`.  Using Equation
    2 (using base 2):

    .. math::
        
        H(X) &= p\log_2(1/p) + (1-p)\log_2(1/p) \\
             &= \log_2(1/p) \\
             &= \log_2(1/0.5) \\
             &= 1 \tag{3}

    So one bit of information is transmitted with every observation of a fair coin toss.
    If we vary the value of :math:`p`, we get a symmetric curve shown in Figure
    1.  The more biased towards H or T, the less entropy (information/surprise)
    we get.

    .. figure:: /images/binary_entropy.png
       :height: 300px
       :alt: Entropy with varying :math:`p` (source: Wikipedia)
       :align: center

       Figure 1: Entropy with varying :math:`p` (source: Wikipedia)

A continuous analogue to (discrete) entropy is called *differential entropy*
(or continuous entropy).  It has a very similar equation using integrals
instead of sums:

.. math::

    H(X) := - \int_{-\infty}^{\infty} f(x)ln(f(x)) dx \tag{4}

We have to be careful with differential entropy because some of the properties
of (discrete) entropy do not apply to differential entropy, for example,
differential entropy can be negative.

|h2| Principle of Maximum Entropy |h2e|



|h2| Further Reading |h2e|

* Wikipedia: `Maximum Entropy Probability Distribution <https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution>`_, `Principle of Maximum Entropy <https://en.wikipedia.org/wiki/Principle_of_maximum_entropy>`_, `Entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_, `Self-Information <https://en.wikipedia.org/wiki/Self-information#Definition>`

|br|

.. [1] This isn't exactly right because beyond the letter frequencies, we also can predict what the word is, which will change the information and entropy.  Natural language also has redundencies such as "q must always be followed by u", so this will change our probability distribution.  See `Entropy and Redundancy in English <http://people.seas.harvard.edu/~jones/cscie129/papers/stanford_info_paper/entropy_of_english_9.htm>`_ for more details.

