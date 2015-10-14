.. title: Probability as Extended Logic
.. slug: probability-the-logic-of-science
.. date: 2015-10-03 23:30:05 UTC-04:00
.. tags: probability, Jayne, logic, mathjax
.. category: 
.. link: 
.. description: Probability as extended logic.
.. type: text

.. |br| raw:: html

   <br/>

.. |H2| raw:: html

   <br/><h3>

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

.. TEASER_END

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
will we assign it a strict true or false truth value, we insead want to assign
it a degree of *plausibility* (of truth).  One way to accomplish this is to
classify a proposition like "it is raining" with a number indicating how
plausibile you think that it is currently raining.  Along with this new method
of evaluating propositions, we'd also like to develop a system to reason about
them while ideally still maintaining the same type of deductive reasoning we
have with Boolean logic, but extending it to handle our new concept of degrees
of plausibility about these propositions. 


|h3| Weaker Rules of Inference |h3e|

We already saw two forms of inference from Boolean logic, Rule R1 and R2:

.. math::

    \text{if }A\text{ is true, then }B\text{ is true}

    \frac{A\text{ is true}}{\text{therefore, }B\text{ is true}}  \tag{R1}

    \frac{B\text{ is false}}{\text{therefore, }A\text{ is false}} \tag{R2}

These rules extend quite naturally to our degress of plausibility.
For R1, if we think that A is plausible (to some degree), then 
it intuitively makes sense that B becomes more plausible.
Similarly for R2, if we think B is implausible (to some degree), then 
A should also become more implausible.  
Using this line of reasoning, we can come up with some more rules of inference
that, while in Boolean logic would be non-sensical, they do make sense in our
new system reasoning with plausibilities.  Consider these new rules R3 and R4:

.. math::

    \text{if }A\text{ is true, then }B\text{ is true}

    \frac{B\text{ is true}}{\text{therefore, }A\text{ is more plausible}}  \tag{R3}

    \frac{A\text{ is false}}{\text{therefore, }B\text{ is less plausible}} \tag{R4}

If we try to apply it to our example above, it passes our simplest smoke test
of a rational line of reasoning:

.. math::

    \text{if it is raining, then it is cloudy}

    \frac{\text{it is cloudy}}{\text{therefore, it is more plausible that it is raining}}

    \frac{\text{it is not raining}}{\text{therefore, it is less plausible that it is cloudy}}

Here, if it's cloudy, we're not positive that it's raining but somehow it has increased
our belief that it will rain ("Is it going to rain?  It might, it looks cloudy.").
Alternatively, if it's not raining there is definitely some degree of plausibility
that it is not cloudy.  With Boolean logic and it's strict true/false
dichotomy, we cannot really make any conclusions but with plausible reasoning we 
can at least make *some* statements.

Of course, there is not much precision (read: mathematics) in what we've said,
we're just trying to gain some intuition on how we would ideally reason about
propositions with varying degress of plausibility.  In whatever system we end
up designing, we'd like to keep the spirit of R1-R4 in tact because it follows
what we would expect a smart rational person to conclude.

|h3| Introducing the Robot |h3e|

In all of the above discussion about plausible reasoning, we've been trying to
build "a mathematical model of human common sense" as Jayne puts it.  However,
we need to be careful because human judgement has many properties (that while
useful) may not be ideal for us to include in our system of reasoning such as
emotion and misunderstandings.  Here is where Jayne introduces a really neat
concept, the robot, in order to make it clear what we're trying to achieve:

    In order to direct attention to constructive things and away from
    controversial irrelevancies, we shall invent an imaginary being.  Its brain
    is to be designed *by us*, so that it reasons according to certain definite
    rules.  These rules will be deduced from simple desiderata which, it
    appears to us, would be desirable in human brains; i.e. we think that a
    rational person, on discovering that they were violating one of these
    desiderata, would wish to revise their thinking. 
    ...
    To each proposition about which it reasons, our robot must assign some
    degree of plausibility, based on the evidence we have given it; and
    whenever it recieves new evidence it must revise these assignments to take
    that new evidence into account.

Sounds like a pretty cool robot!  So then our goal is not to build this
hypothetical robot that follows certain rules (which we'll define) and
be consistent with what how an ideal rational person would reason.
Here is a list of his three requirements (desiderata):

 1. Degrees of plausibility are represented by real numbers.
 2. Qualitative correspondence with common sense.
 3. Consistency:
 
    a. If a conclusion can be reasoned out in more than one way, then every possible way must lead to the same result.
    b. The robot always takes into account all of the evidence it has relevant to the question.  It does not artbitrarily, ignore some of the information, basing its conclusions only on what remains.  In other words, the robot is nonideological.
    c. The robot always represents equivalent states of knowledge by equivalent plausibility assignments.  That is, if in two problems the robot's state of knowledge is the same (except perhaps for the labeling of the propositions), then it must assign the same plausibilities in both.  

The first requirement is mostly for practicality.  If we're building a machine,
we'd like some standard way to tell it about plausibility (and vice versa),
real numbers seem appropriate.
The second requirement tells us that the robot should at least qualitatively
reason like humans do.  For example, the robot should be able to reason
somewhat like our rules R1-R4 above, which is precisely the whole point of our
exercise. 
The last requirement is obvious since if we're trying to build a robot
to reason, it has to be consistent (or what use is it?).

What is surprising is that from these three desiderata, Jayne goes on
to derive probability theory (extending it from Boolean logic)!  If you're
interested, I encourage you to check out his book `Probabilty Theory: The Logic
of Science <http://bayes.wustl.edu/etj/prob/book.pdf>`_ (first three chapters
online), where in Chapter 2 he goes over all the gory details.  It's quite an
interesting read and pretty accessible if you know a bit of calculus and are
comfortable with some algebraic manipulation.  I'll spare you the details here
on how the derivation plays out (as I'm probably not the right person to
explain it) but instead I want to focus on next is the result of how
probability theory can be viewed as an extension of Boolean logic.

|h2| Probability as Extended Logic |h2e|

The rules of probability have direct analogues with our Boolean operators above
(as it can be viewed as an extension).
Now our propositions don't have 0 or 1 truth values, they can take on any value
in the range 0 (false) to 1 (true) representing their plausibility.  The symbol
:math:`P(A|B)` is used to denote the degree of plausibility we assign
proposition A, given our background or prior knowledge B (remember the robot
will take all relevant known information into account).

The really interesting insight is that all the concepts from Boolean logic are
just limiting cases of our extension (i.e. probability theory) where our robot
becomes more and more cetain of itself.  Let's take a look.

|h3| Extended Boolean Operators |h3e|

Consider negation ("not" operator).  The analogue in probability theory is the
basic sum rule:

.. math::

    P(A|B) + P(\bar{A}|B) = 1

If we are entirely confident in proposition A (:math:`P(A|B)=1` or A is true),
then from the above rule, we can conclude :math:`P(\bar{A}|B) = 1 - P(A|B) = 0`,
or :math:`\bar{A}` is false.

This works equally well with our two basic Boolean operators.  Consider the "and"
operator, it's analogue is the product rule:

.. math::

    P(AB|C) = P(A|BC)P(B|C) = P(B|AC)P(A|C)

Let's try a few cases out.  If A is true and B is true, we should see that AB
is true.  Translating that to probabilities, we get :math:`P(A|C)=1` and
:math:`P(B|C)=1`.  Now this doesn't fit as nicely into our product rule
but we just need to go back to the concept of our robot taking all known
information into account.  If we know that :math:`P(B|C)=1`, this means
that given background information :math:`C`, we know enough to conclude
that :math:`B` is plausible with absolute certainty.  If we then add
additional background information that A is also plausible with absolute
certainty (given the same background information), then we can conclude that
:math:`P(B|AC)=1` because A is no longer relevant and our robot only uses
:math:`C` as the relevant background information when computing the
plausibility of :math:`B` [1]_.  Plugging it into the formula we get the
desired result of :math:`P(AB|C)=1`.  And since "and" operator is commutative,
we could have easily used the second expression and reached the same
conclusion.
Alternatively, if we try :math:`P(A|C)=0` and :math:`P(B|C)=1`, we can see
through a similar line of reasoning that the result should be :math:`P(AB|C)=0`.

The final Boolean operator "or" also has a direct analogue in the extended sum
rule:

.. math::

    P(A + B|C) = P(A|C) + P(B|C) - P(AB|C)

Taking a similar line of reasoning, if we have :math:`P(A|C)=0` and
:math:`P(B|C)=1`, we have :math:`P(AB|C)=0` from the above line of reasoning.
With these three quantities, we can easily compute :math:`P(A + B|C)=1`, as we
would expect (If A is false and B is true, then "A or B" is true).

|h3| Extended Reasoning |h3e|

As we saw before, we would ideally like our original rules (R1 and R2) as well
as our extended rules (R1-R4) to be included in our new system.
As expected, these common sense interpretations are preserved in probability
theory with a modified form of the product rule.

Recall the rules R1 and R2:

.. math::

    \text{if }A\text{ is true, then }B\text{ is true}

    \frac{A\text{ is true}}{\text{therefore, }B\text{ is true}}  \tag{R1}

    \frac{B\text{ is false}}{\text{therefore, }A\text{ is false}} \tag{R2}

The premise can be encoded in our background information :math:`C`:

.. math::

    C \equiv A \implies B

Given this background information, we can use these forms of the product rule to encode
R1, R2 as rules PR1 and PR2, respectively:

.. math::

    P(B|AC) = \frac{P(AB|C)}{P(A|C)}                    \tag{PR1} \\
    P(A|\bar{B}C) = \frac{P(A\bar{B}|C)}{P(\bar{B}|C)}  \tag{PR2}

This is not all that obvious because we lose some of the nice one-to-one correspondence
like the operators above.  However, treating A, B, C as propositions, aids us in decoding
these equations.  Given our major premise :math:`C \equiv A \implies B`, let's
look at the truth table for the relevant propositions.  

.. table::

    =====  =====  =============================  ==============  ========
      A      B    :math:`C \equiv A \implies B`  :math:`AB | C`  :math:`A\bar{B}|C`
    =====  =====  =============================  ==============  ========
    False  False  True                           False           False
    False  True   True                           False           False
    True   False  False                          *Impossible*    *Impossible*
    True   True   True                           True            False
    =====  =====  =============================  ==============  ========

Notice that this truth table is a bit special in that I am mixing our extended logic
with Boolean logic (e.g. :math:`|` symbol).  Although it's not really proper to
do so, this is more an exercise in intuition than anything else so I'll stick
with the sloppiness for sake of explanation.
Next, we see that I have filled in a special notation for the third row
using the term "*impossible*".  This is to indicate, given the premise :math:`C`,
this situation cannot possibly occur (or else our premise would be false).

Now given this truth table, we can see that :math:`AB | C` simplifies to
the expression :math:`A` by ignoring the impossible row from our premise.
Similarly, :math:`A\bar{B}|C` simplifies to "False".  Plugging these back
into PR1 and PR2:

.. math::

    P(B|AC) = \frac{P(AB|C)}{P(A|C)} = \frac{P(A|C)}{P(A|C)} = 1.0  \\
    P(A|\bar{B}C) = \frac{P(A\bar{B}|C)}{P(\bar{B}|C)} = \frac{0}{P(\bar{B}|C)} = 0.0

we get the desired result.  In particular, :math:`P(B|AC)` resolves to the same
thing that :math:`A \implies B, A` resolves to: :math:`B` is true.  Similarly,
:math:`P(A|\bar{B}C)` resolves to the same thing that :math:`A \implies B,
\bar{B}` resolves to: :math:`A` is false.  Pretty neat, huh?


The rules R3 and R4 also extend quite naturally from our product rule.  Recall
rules R3 and R4:

.. math::

    \text{if }A\text{ is true, then }B\text{ is true}

    \frac{B\text{ is true}}{\text{therefore, }A\text{ is more plausible}}  \tag{R3}

    \frac{A\text{ is false}}{\text{therefore, }B\text{ is less plausible}} \tag{R4}

R3 can be encoded as the product rule:

.. math::

    P(A|BC) = P(A|C)\frac{P(B|AC)}{P(B|C)}

But from the discussion above, we know :math:`P(B|AC)=1` and 
:math:`P(B|C) \leq 1` (from the definition of a probability), so it must be the
case that:

.. math::

    P(A|BC) \geq P(A|C)  \tag{E1}

In other words, given new information :math:`B`, we now think :math:`A` is more
plausible. We can build upon this to reason about R4 using this form of the
product rule:

.. math::

    P(B|\bar{A}C) = P(B|C)\frac{P(\bar{A}|BC)}{P(\bar{A}|C)}

From E1, we know that :math:`P(\bar{A}|BC) \leq P(\bar{A}|C)` (remember
the "not" rule), so we can conclude that:

.. math::

    P(B|\bar{A}C) \leq P(B|C)

which says that given :math:`\bar{A}`, proposition :math:`B` becomes less
plausible.

|h2| Conclusion |h2e|

Some conclusion...


|h2| Further Reading |h2e|

* `Probabilty Theory: The Logic of Science <http://bayes.wustl.edu/etj/prob/book.pdf>`_ (first three chapters) by E. T. Jayne.
* `Probability Theory As Extended Logic <http://bayes.wustl.edu/>`_ at Washington University In St Louis.
* `Probability, Paradox, and the Reasonable Person Principle <http://nbviewer.ipython.org/url/norvig.com/ipython/Probability.ipynb>`_ by Peter Norvig

|br|

.. [1] You might wonder what happens when :math:`A` and :math:`C` are mutually exclusive propositions (i.e. impossible to happen at the same time).  In this case, :math:`P(B|AC)` is not defined but also our original question is ill formed because we couldn't have the case :math:`P(A|C)=1` (we would instead have :math:`P(A|C)=1`).
