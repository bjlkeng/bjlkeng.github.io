.. title: Optimal Betting Strategies and The Kelly Criterion
.. slug: optimal-betting-and-the-kelly-criterion
.. date: 2015-11-03 19:13:31 UTC-05:00
.. tags: betting, Kelly Criterion, probability, mathjax
.. category: 
.. link: 
.. description: 
.. type: A look at optimal betting and the Kelly criterion digging into some of the math.

.. |br| raw:: html

   <br />

.. |H2| raw:: html

   <br><h3>

.. |H2e| raw:: html

   </h3>

.. |H3| raw:: html

   <h4>

.. |H3e| raw:: html

   </h4>


My last post was about some `common mistakes
<link://slug/gamblers-fallacy-and-the-law-of-small-numbers>`_ when betting
(gambling) even with an understanding of probability.  This post is going to
talk about the other side: optimal betting strategies using some very
interesting results from some very famous mathematicians in the 50s and 60s. 
I'll spend some time digging through some of the math, introducing some new
concepts (at least to me), seting the problem up and doing a quick sketch of
the proof.  We'll be looking at it from the lens of our simplest probability
problem: the coin flip.  As usual, I will not be covering the part that shows
you how to make a fotune -- that's an exercise best left to the reader.

.. TEASER_END

|h2| Background |h2e|

|h3| History |h3e|

There is an incredibly facinating history surrounding the development of the
mathematics around gambling and optimal betting strategies.  The optimal
betting strategy, more commonly known as the `Kelly Criterion
<https://en.wikipedia.org/wiki/Kelly_criterion>`_, was developed in the 50s by
`J. L. Kelly <http://home.williampoundstone.net/Kelly.htm>`_ , a scientist
working at Bell Labs on data compression schemes at the time.  In 1956, he made
an ingenious connection between, his colleague's (`Shannon
<https://en.wikipedia.org/wiki/Claude_Shannon>`_) work on information theory,
gambling, and a television game show publishing his new findings in a paper
titled *A New Interpretation of Information Rate* (whose original title was
*Information Theory and Gambling*).  

The paper remained unnoticed until the 1960s when an MIT student named Ed Thorp
told Shannon about his card-counting scheme to beat blacjack.  Kelly's paper
was referred to him, and Thorp started using it to amass a small fortune using
Kelly's optimal betting strategy along with his card-counting system.  Thorp
and his colleagues later went on to use the Kelly Criterion in other
varied gambling applications such as horse racing, sports betting, and even the
stock market.  Thorp's hedge fund outperformed many of his peers and it was
this success that Wall Street take notice of the Kelly Criterion.  There is a
great book called Fortune's Formula [1]_ that details the stories and
adventures surrounding these brilliant minds.

|h3| Surely, Almost Surely |h3e|

In probability theory, there are two terms that distinguish very similar
conditions: `sure and almost sure <https://en.wikipedia.org/wiki/Almost_surely#.22Almost_sure.22_versus_.22sure.22>`_.
If an event is **sure**, then it always happens.  That is, it is not possible for
any other outcome to occur.  If an event is **almost sure** then it occurs with
probability 1.  That is, theoretically there is an outcomes not belonging to
this event that can occur, but the probability is so small that it's smaller
than any fixed positive probability, and therefore must be 0.  This is kind of
abstract, so let's take a look at an example (from `Wikipedia <https://en.wikipedia.org/wiki/Almost_surely>`_).

Imagine we have a unit square where we're randomly throwing point-sized darts that
will land inside the square with a uniform distribution.  For the entire square
(light blue), it's easy to see that it makes up the entire sample space, so we would
say that the dart will *surely* land within the unit square because there is no
other possible outcome.

.. image:: /images/unit_square.png
   :height: 350px
   :alt: unit square
   :align: center

Further, the probability of landing in any given region is the ratio of its
area to the ratio of the total unit square, simplifying to just the area of a
given region.  For example, taking the top left corner (dark blue), which
is 0.5 units x 0.5 units, we could conclude that :math:`P(\text{dart lands in
dark blue region}) = (0.5)(0.5) = 0.25`.

Now here's the interesting part, notice that there is a small red dot in the
upper left corner.  Imagine this is just a single point at the upper left
corner on this unit square.  What is the probability that the dart lands on the
red dot?  Since the red dot has an area of :math:`0`, :math:`P(\text{dart lands
on red dot}) = 0`.  So we could say that the dart *almost surely* does not land
on the red dot.  That is, theoretically it could, but the probability of doing
so is :math:`0`.  The same argument can be made for *every* point in the sqaure.  

The dart actually does land on a single point of the square though, so even
though the probability of landing on that point is :math:`0`, it still does
occur.  For these situations, it's not *sure* that we won't hit that specific
point but it's *almost sure*.  A subtle difference but quite important one.



|br|
|br|

.. [1] William Poundstone, *Fortune's Formula: The Untold Story of the Scientific Betting System That Beat the Casinos and Wall Street*. 2005. ISBN 978-0809045990.  See also a brief `biography <http://home.williampoundstone.net/Kelly.htm>`_ of Kelly on William Poundstone's web page.

