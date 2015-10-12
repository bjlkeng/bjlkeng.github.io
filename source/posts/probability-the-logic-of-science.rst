.. title: Probability as Extended Logic
.. slug: probability-the-logic-of-science
.. date: 2015-10-03 23:30:05 UTC-04:00
.. tags: probability, Jayne, logic, mathjax
.. category: 
.. link: 
.. description: Probability as extended logic.
.. type: text

.. |br| raw:: html

   <br />

.. |H2| raw:: html

   <h3>

.. |H2e| raw:: html

   </h3>

.. |H3| raw:: html

   <h4>

.. |H3e| raw:: html

   </h4>


Modern probability theory is typically derived from the
`Kolmogorov axioms <https://en.wikipedia.org/wiki/Probability_axioms>`_,
using measure theory with the notions of events and sample space.
In one way, it's intuitive to understand how this works as Laplace 
`wrote <https://en.wikipedia.org/wiki/Classical_definition_of_probability>`_:

    The probability of an event is the ratio of the number of cases favorable
    to it, to the number of all cases possible, when [the cases are] equally
    possible. ... Probability is thus simply a fraction whose numerator is the
    number of favorable cases and whose denominator is the number of all the
    cases possible.

However, the intuition of this view of probabilty breaks down when we want to
do more complex reasoning.  After learning probabily from the lens of coins,
dice and urns full of red and white balls, I still didn't feel that I had
have a strong intuition about how to apply it to other situtions -- especially
ones where it was difficult or too abstract to apply the idea of *"a fraction
whose numerator is the number of favorable cases and whose denominator is the
number of all the cases possible"*.  And then I read `Probabily Theory: The Logic of Science <http://www.cambridge.org/gb/academic/subjects/physics/theoretical-physics-and-mathematical-physics/probability-theory-logic-science>`_ by E. T. Jayne.

Jayne takes a drasticlly different view of probability, not with events and sample spaces,
but rather an extension of Boolean logic.  Taking this view made a great deal of sense
to me since I spent a lot of time `studying and reasoning
<link://slug/accessible-satisfiability>`_ in Boolean logic.  The following post
is my attempt to explain Jayne's view of probability theory, where he derives
it from "common sense" extensions to Boolean logic.  (*Spoiler alert: he ends
up with pretty much the same mathematical system as Kolmogorov's probability theory.*)
I'll stay away from any heavy derivations and stick with the intuition, which
is exactly why I think this view of probability theory is more useful.

|h2| Boolean Logic |h2e|

*Note: Feel free to skip this section if you're already comfortable with Boolean logic.*

Before we begin with probability, let's do a quick review of Boolean logic
(sometimes also called propositional logic or propositional calculus).
In the context of modeling real-world situations, we usually define
propositions to describe things we may want to reason about,
denoted by :math:`\{A, B, C \ldots\}`.  Propositions have an unambiguous
meaning, and must either true or false.  For example the following two
sentences could be propositions:

.. math::

    A &:= \text{It is raining.} \\
    B &:= \text{It is cloudy.}

We could also define a logical relation between the two propositions 
using an implication operator (colloquially if-then statement):

.. math::

    \text{if }A\text{ is true, then }B\text{ is true} := \text{if it is raining, then it is cloudy}

|h3| Rules of Inference |h3e|

To reason about them, we usually use two forms of inference, `modus ponens
<https://en.wikipedia.org/wiki/Modus_ponens>`_ (Rule R1):

.. math::

    \text{if }A\text{ is true, then }B\text{ is true}

    \frac{A\text{ is true}}{\text{therefore, }B\text{ is true}}  \tag{R1}

and also `modus tollens <https://en.wikipedia.org/wiki/Modus_tollens>`_ (Rule R2):

.. math::

    \text{if }A\text{ is true, then }B\text{ is true}

    \frac{B\text{ is false}}{\text{therefore, }A\text{ is false}} \tag{R2}

Both make intuitive sense when you try to apply it to examples like the one above:

.. math::

    \text{if it is raining, then it is cloudy}

    \frac{\text{it is raining}}{\text{therefore, it is cloudy}}

and:

.. math::

    \text{if it is raining, then it is cloudy}

    \frac{\text{it is not cloudy}}{\text{therefore, it is not raining}}

|h3| Basic Boolean Operators |h3e|

There are several Boolean operators that are pretty natural things to do
when discussing propositions.  The most basic on is the **negation** (or "not")
operator, usually drawn with a bar above the proposition (or expression):

.. math::

    \bar{A}

The next one is **conjunction** (or the "and"
operator) meaning "both A and B are true", denoted by:

.. math::

    AB

The final one is **disjunction** (or the "or"
operator) meaning "at least one the propositions A, B are true", denoted with a "+" sign:

.. math::

    A + B

With the above examples, our intuition isn't too far off from the natural
English interpretation (except for "or", which is inclusive instead of
exclusive usually used in English):

.. math::

    \bar{A} &:= \text{it is }\textbf{not}\text{ raining} \\
    AB &:= \text{it is raining }\textbf{and}\text{ it is cloudy} \\
    A + B &:= \text{it is raining }\textbf{or}\text{ it is cloudy (or both)}

|h3| Limitations of Boolean Logic |h3e|

Boolean logic has wide applications in many areas.  It is one of the
fundamental ideas used in modern computing and one of the simplest symbolic
logic systems in modern use.  From one point of view, it's quite a natural
way to rationally reason about real-world problems.  With repeated
applications of Rules R1 or R2, we can logically "prove" a fact from a set of
premises.  In fact, this type of reasoning system has been used for centuries 
with `Aristotelian logic <https://en.wikipedia.org/wiki/Term_logic>`_.
However, it's not too hard to see that it has some limitations on the kinds of
things that can be modeled with it.

For example, given our above proposition "it is raining", using Boolean logic,
we would have to assign this an "unambiguous meaning... either true or false".
If we think a bit, we can probably come up with a situation where it's not so
clear whether the statement should be clearly true or false.  
Perhaps I'm in my bedroom and my curtains are closed but I can see that it
looks kind of grey outside.  Am I 100% certain that it is raining, or is there
more of a 50/50 chance that it is raining.  Clearly, Boolean logic isn't quite
ready to handle these situations but if we somehow relaxed the criteria so
that each proposition didn't have to be 100% true or 100% false, we could
come up with a reasoning system that could be used to model a wider variety of
real-world situations.  In the next section, we'll introduce some ideas to get
us closer to this type of system.

|h2| Plausible Reasoning |h2e|

We've just seen that one of the big limitations to Boolean logic is the strict
true or false values that we need to assign to propositions.  If we try to
relax this constraint a bit, we end up with something that can model quite a
few more situations.  For a proposition such as "it is raining".  No longer
will we assign it a strict true or false, we insead want to assign it a degree
of *plausibility*.  One way to accomplish this is to classify a proposition like
"it is raining" with a number indicating how plausibile you think that it is
currently raining.  Along with this new method of evaluating propositions,
we'd also like to develop a system to reason about them, ideally still
maintaining the same type of deductive reasoning we have with Boolean logic,
but extended to reasonably handle our degrees of plausibility about these new
propositions. 


|h3| Weaker Rules of Inference |h3e|



To reason about them, we usually use two forms of inference, `modus ponens
<https://en.wikipedia.org/wiki/Modus_ponens>`_ (Rule R1):

.. math::

    \text{if }A\text{ is true, then }B\text{ is true}

    \frac{A\text{ is true}}{\text{therefore, }B\text{ is true}}  \tag{R1}

and also `modus tollens <https://en.wikipedia.org/wiki/Modus_tollens>`_ (Rule R2):

.. math::

    \text{if }A\text{ is true, then }B\text{ is true}

    \frac{B\text{ is false}}{\text{therefore, }A\text{ is false}} \tag{R2}











|h2| Introducing the Robot |h2e|


Modeling 


Historically, 

t



.. TEASER_END

|h2| Further Reading |h2e|

* `Probability, Paradox, and the Reasonable Person Principle <http://nbviewer.ipython.org/url/norvig.com/ipython/Probability.ipynb>`_ by Peter Norvig
* `Probability Theory As Extended Logic <http://bayes.wustl.edu/>`_ at Washington University In St Louis.
* `Probabilty Theory: The Logic of Science <http://bayes.wustl.edu/etj/prob/book.pdf>`_ (first three chapters) by E. T. Jayne.

|br|

.. [1]
