.. title: Lagrange Multipliers
.. slug: lagrange-multipliers
.. date: 2016-11-07 07:48:31 UTC-05:00
.. tags: lagrange multipliers, calculus, mathjax
.. category: 
.. link: 
.. description: A quick primer on lagrange multipliers.
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

This post is going to talk about finding the maxima or minima of a function
subject to some constraints.  This is usually introduced in a multivariate
calculus course, unfortunately (or fortunately?) I never got the chance to take
a multivariate calculus course that covered this topic.  In my year, computer
engineers only took three half year engineering calculus courses, and the
`fourth one <http://www.ucalendar.uwaterloo.ca/1617/COURSE/course-ECE.html#ECE206>`_ 
(for electrical engineers) seems to have covered other basic multivariate
calculus topics such as all the various theorems such as Green's, Gauss', Stokes' (I
could be wrong though, I never did take the course!).  As I imagine Newton
always said, "It's never too late to learn multivariate calculus!".

In that vein, this post will discuss one widely used method for finding optima
subject to constraints: Lagrange multipliers.  The concepts
behind it are actually quite intuitive (once we come up with the right analogue
in physical reality) so as usual we'll start there.  We'll work through some
problems and hopefully by the end of this post, this topic won't seem as
mysterious anymore [1]_.

|h2| Motivation |h2e|

One of the most common problems of calculus is finding the minima or maxima of
a function.  We see this in a first year calculus course where we take the
derivative, set it to :math:`0`, and solve for :math:`x`.  Although this covers
a wide range of applications, there are *many* more interesting problems that
come up that don't fit into this simple mold.  

A few interesting examples:

1. A simple (and uninteresting) question might be: "How do I minimize the
   aluminum (surface area) I need to make this can?"  It's quite simple: just
   make the can really, really small!  A more useful question might be: "How do
   I minimize the aluminum I need to make this can, assuming that I need to
   hold 250ml of liquid?"
2. `Milkmaid problem <http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html>`_:
   Suppose we're a milkmaid (:math:`M`) in a large flat field trying to get to
   our cow (:math:`C`) as fast as we can (so we can finish milking it and get
   back to watching Westworld).  Before we can milk the cow, we have to clean
   out our bucket in the river nearby (defined by :math:`g(x,y)=0`).  The
   question becomes what is the best point on the river bank (:math:`P`) to
   minimize the total distance we need to get to the cow?  We want to minimize
   distance here but clearly we also have the constraint that we must hit the
   river bank first (otherwise we would just walk in a straight line).
3. Finding the maximum likelihood estimate (MLE) of a multinomial distribution
   given some observations.  For example, suppose we have a weighted six-sided
   die and we observe rolling it :math:`n` times (i.e. we have the count of the
   number of times each side of the die has come up).  What is the MLE estimate
   for each of the die probabilties (:math:`p_1, \ldots, p_6`?  It's clear
   we're maximizing the likelihood function but we're also subject to the constraint
   that the probabilities need to sum to :math:`1`, i.e. :math:`\sum_{i=1}^6 p_i = 1`.

We'll come back to these and try to solve them a bit later on.  

The reason why these problems are more interesting is because many real-world
impose constraints on what we're trying to minimize/maximize (reality can be
pretty annoying sometimes!).  Fortunately, Lagrange multipliers can help us in
all three of these scenarios.





.. TEASER_END

|h2| Further Reading |h2e|

* Wikipedia: `Lagrange multiplier <https://en.wikipedia.org/wiki/Lagrange_multiplier>`_
* `An Introduction to Lagrange Multipliers <http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html>`_, Steuard Jensen

|br|

.. [1] This post draws heavily on a great tutorial by Steuard Jensen: `An Introduction to Lagrange Multipliers <http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html>`_.  I highly encourage you to check it out.
