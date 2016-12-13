.. title: Lagrange Multipliers
.. slug: lagrange-multipliers
.. date: 2016-12-13 07:48:31 UTC-05:00
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

This post is going to be about finding the maxima or minima of a function
subject to some constraints.  This is usually introduced in a multivariate
calculus course, unfortunately (or fortunately?) I never got the chance to take
a multivariate calculus course that covered this topic.  In my undergraduate class, computer
engineers only took three half year engineering calculus courses, and the
`fourth one <http://www.ucalendar.uwaterloo.ca/1617/COURSE/course-ECE.html#ECE206>`_ 
(for electrical engineers) seems to have covered other basic multivariate
calculus topics such as all the various theorems such as Green's, Gauss', Stokes' (I
could be wrong though, I never did take that course!).  You know what I always imagined Newton
saying, "It's never too late to learn multivariate calculus!".

In that vein, this post will discuss one widely used method for finding optima
subject to constraints: Lagrange multipliers.  The concepts
behind it are actually quite intuitive once we come up with the right analogue
in physical reality, so as usual we'll start there.  We'll work through some
problems and hopefully by the end of this post, this topic won't seem as
mysterious anymore [1]_.

.. TEASER_END

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
   I minimize the aluminum (surface area) I need to make this can, assuming
   that I need to hold :math:`250 cm^3` of liquid?"
2. `Milkmaid problem <http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html>`_:
   Suppose we're a milkmaid (:math:`M`) in a large flat field trying to get to
   our cow (:math:`C`) as fast as we can (so we can finish milking it and get
   back to watching Westworld).  Before we can milk the cow, we have to clean
   out our bucket in the river nearby (defined by :math:`g(x,y)=0`).  The
   question becomes what is the best point on the river bank (:math:`(x,y)`) to
   minimize the total distance we need to get to the cow?  We want to minimize
   distance here but clearly we also have the constraint that we must hit the
   river bank first (otherwise we would just walk in a straight line).
3. Finding the maximum likelihood estimate (MLE) of a multinomial distribution
   given some observations.  For example, suppose we have a weighted six-sided
   die and we observe rolling it :math:`n` times (i.e. we have the count of the
   number of times each side of the die has come up).  What is the MLE estimate
   for each of the die probabilities (:math:`p_1, \ldots, p_6`)?  It's clear
   we're maximizing the likelihood function but we're also subject to the constraint
   that the probabilities need to sum to :math:`1`, i.e. :math:`\sum_{i=1}^6 p_i = 1`.

We'll come back to these and try to solve them a bit later on.  

The reason why these problems are more interesting is because many real-world
impose constraints on what we're trying to minimize/maximize (reality can be
pretty annoying sometimes!).  Fortunately, Lagrange multipliers can help us in
all three of these scenarios.

|h2| Lagrange Multipliers |h2e|

|h3| The Basics |h3e|

Let's start out with the simplest case in two dimensions since it's easier to
visualize.  We have a function :math:`f(x, y)` that we want to maximize and
also a constraint :math:`g(x, y)=0` that we must satisfy.  Translating this into a
physical situation, imagine :math:`f(x, y)` defines some hill.  At each
:math:`(x,y)` location, we have the height of the hill.  Figure 1 shows
this as a blue (purple?) surface.  Now image we draw a red path (as shown in
Figure 1) on our :math:`(x, y)` plane, defined by :math:`g(x, y)=0`.  We wish
to find the highest point on the hill that stays on this red path (i.e.
maximize :math:`f(x,y)` such that :math:`g(x,y)=0`).

.. figure:: /images/450px-LagrangeMultipliers3D.png
   :height: 300px
   :alt: Lagrange Multipliers 3D (source: Wikipedia)
   :align: center

   Figure 1: A 3D depiction of :math:`f(x,y)` (blue surface) and :math:`g(x,y)`
   (red line). (source: Wikipedia)

If we imagine ourselves trying to solve this problem in the physical world, there is (at
least) one easy way to go about it: Keep walking along the red path, which
potentially will take you up and down the hill, until you find the highest
point (:math:`f(x,y)` maximized).  This brute force approach may be a simple way to
achieve our goal but remember that :math:`g(x,y)=0` might be infinitely long so
it's a bit less practical.  Instead, we'll do something a bit less exhaustive
and identify candidate points along the path that *may* end up being maxima
(i.e. necessary conditions).

Using our original idea of walking along the path, how do we know when we've
potentially come across a candidate maxima point?  Well, if we are walking uphill for a while
along the path and then suddenly start walking downhill.  If this sounds
familiar, this resembles the idea of setting derivative to :math:`0`.
In higher dimensions the generalization of the derivative is a gradient but
we're not simply setting it to :math:`0` because, remember, we're actually
walking along the :math:`g(x,y)` path.

.. admonition:: Gradients

   The `gradient <https://en.wikipedia.org/wiki/Gradient>`_, denoted with the
   :math:`\nabla` symbol, defines a vector of
   the :math:`n` partial derivatives of a function of several variables:
   :math:`\nabla f(x,y) = (\frac{\partial f(x,y)}{\partial x}, \frac{\partial
   f(x,y)}{\partial y})`.  Similar to a single variable function, the direction
   of the gradient points in the direction of the greatest rate of increase and its
   magnitude is the slope in that direction.
   
   By this `theorem <https://en.wikipedia.org/wiki/Level_set#Level_sets_versus_the_gradient>`_,
   a :math:`\nabla f`'s direction is either zero or perpendicular to contours 
   (points at the same height)
   of :math:`f`.  Using the analogy from Wikipedia, if you have two hikers at the same
   place on our hill.  One hikes in the direction of steepest ascent uphill
   (the gradient).  The other one is more timid and just walks along the hill
   at the same height (contour lines).  The theorem says that the two hikers
   will (initially) walk perpendicular to each other.

There are two cases to consider if we've walked up and down the hill while staying
along our path.  Either we've reached the top of the hill and the gradient of
:math:`f(x,y)` will be zero.  Or while walking along the path :math:`g(x,y)=0`,
we have reached a level part of :math:`f(x,y)` which means our :math:`f`
gradient is perpendicular to our path from the theorem in the box above.  But
since for this particular point, our path follows the level part of :math:`f`,
the gradient of :math:`g` at this point is also perpendicular to the path.
Hence, the two gradients are pointing in the same direction.  Figure 2 shows a
depiction of this idea in 2D, where blue is our hill, red is our path and the
arrows represent the gradient.

.. figure:: /images/450px-LagrangeMultipliers2D.png
   :height: 300px
   :alt: Lagrange Multipliers 2D (source: Wikipedia)
   :align: center

   Figure 2: A 2D depiction of :math:`f(x,y)` and :math:`g(x,y)=0`.  The
   points on each ellipses define a constant height.  The arrows define the
   direction of the gradient. (source: Wikipedia)

We can see that at our level point along the path, our red line is tangent to the
contour line of the hill, thus the gradients point in the same direction (up to
a constant multiplier).  This makes sense because our path and the contour line
follow the same direction (at least at that point), which means the gradients
(which are perpendicular to the path) must be equal.  Now there's no guarantee
that the magnitudes are the same, so we'll introduce a constant multiplier
called a **Lagrange Multiplier** to balance them.  Translating that into some
equations, we have:

.. math::

    \nabla f(x,y) &= \lambda \nabla g(x, y) \tag{1} \\
    g(x,y) &= 0 \tag{2}

Equation 1 is what we just described, and Equation 2 is our original condition
on :math:`g(x,y)`.  That's it!  Lagrange multipliers are nothing more than
these equations. You have three equations (gradient in two dimensions has two
components) and three unknowns (:math:`x, y, \lambda`), so you can solve for 
the values of :math:`x,y,\lambda` and find the point(s) that maximizes :math:`f`.
Note that this is a necessary, not sufficient condition.  Generally the solutions
will be `critical points
<https://en.wikipedia.org/wiki/Critical_point_(mathematics)>`_ of :math:`f`, so
you probably will want to find all possible solutions and then pick the max/min
among those ones (or use another method to guarantee the global optima).


|h3| The Lagrangian |h3e|

It turns out solving the solutions for Equation 1 and 2 are equivalent to
solving the maxima of another function called the  *Lagrangian*
:math:`\mathcal{L}`:

.. math::

    \mathcal{L}(x, y, \lambda) = f(x,y) - \lambda g(x, y) \tag{3}

If Equation 3 looks similar to the ones above, it should.  Using the usual method
of finding optima by taking the derivative and setting it to zero, we get:

.. math::

    \nabla_{x, y, \lambda} \mathcal{L}(x, y, \lambda) &= 0 \\
    \Longleftrightarrow \\
    \frac{\partial f(x,y)}{\partial x} - \lambda \frac{\partial g(x,y)}{\partial x} &= 0 \\
    \frac{\partial f(x,y)}{\partial y} - \lambda \frac{\partial g(x,y)}{\partial y} &= 0 \\
    \frac{\partial f(x,y)}{\partial \lambda} - \frac{\lambda \partial g(x,y)}{\partial \lambda} &= 0 \\
    \Longleftrightarrow \\
    \frac{\partial f(x,y)}{\partial x} = &\lambda \frac{\partial g(x,y)}{\partial x} \\
    \frac{\partial f(x,y)}{\partial y} = &\lambda \frac{\partial g(x,y)}{\partial y} \\
    g(x,y) = &0
    \tag{4}

As we can see, with a bit of manipulation these equations are equivalent to
Equations 1 and 2.  You'll probably see both ways of doing it depending on 
which source you're using.

|h3| Multiple Variables and Constraints |h3e|

Now the other non-trivial result is that Lagrange multipliers can extend to any
number of dimensions (:math:`{\bf x} = (x_1, x_2, \ldots, x_n)`) and any number of
constraints (:math:`g_1({\bf x})=0, g_2({\bf x})=0, \ldots, g_m({\bf x}=0)`).
The setup for the Lagrangian is essentially the same thing with one Lagrange multiplier
for each constraint:

.. math::

    \mathcal{L}(x_1, \ldots, x_n, \lambda_1, \ldots, \lambda_n) = 
        f(x_1,\ldots, x_n) - \sum_{k=1}^{M} \lambda_k g_k(x_1, \ldots, x_n) \tag{5}

This works out to solving :math:`n + M` equations with :math:`n + M` unknowns.

|h2| Examples |h2e|

Now let's take a look at solving the examples from above to get a feel for how
Lagrange multipliers work.

.. admonition:: Example 1: Minimizing surface area of a can given a constraint.
    
    **Problem**: Find the minimal surface area of a can with the constraint that its
    volume needs to be at least :math:`250 cm^3`.

    Recall the surface area of a cylinder is:

    .. math::
        
        A(r, h) = 2\pi rh + 2\pi r^2  \tag{6}

    This forms our "f" function.  Our constraint is pretty simple, the volume
    needs to be at least :math:`K=250 cm^3`, using the formula for the volume
    of a cylinder:

    .. math::
        
        V(r, h) &= \pi r^2 h = K \\
        g(r, h) &:= \pi r^2 h - K = 0 \tag{7}

    Now using the method of Lagrange multipliers by taking the appropriate
    derivative, we get the following equations:

    .. math::
        
        \frac{\partial A(r, h)}{\partial r} &= \lambda \frac{\partial V(r, h)}{\partial r} \\
        2\pi h + 4\pi r &= 2 \lambda \pi r h \\
        2r + h(1-\lambda r) &= 0 \tag{8}
    
        \frac{\partial A(r, h)}{\partial h} &= \lambda \frac{\partial V(r, h)}{\partial h} \\
        2\pi r &= \lambda \pi r^2  \\
        r &= \frac{2}{\lambda} \tag{9}

        g(r, h) &= \pi r^2 h - K = 0 \tag{10} \\

    Solving 8, 9 and 10, we get:

    .. math::
    
        \lambda &= \sqrt[3]{\frac{16\pi}{K}} \\
        r &= 2\sqrt[3]{\frac{K}{16\pi}} \\
        h &= \frac{K}{4\pi}(\frac{16\pi}{K})^{\frac{2}{3}}
        \tag{11}

    Plugging in K=250, we get (rounded) :math:`r=3.414, h=6.823`, giving a
    volume of :math:`250 cm^3`, and a surface area of :math:`219.7 cm^2`.


.. admonition:: Example 2: Milkmaid Problem.

    Since we don't have a concrete definition of :math:`g(x,y)`, we'll just
    set this problem up.  The most important part is defining our function to
    minimize.  Given our starting point :math:`P`, our point we hit along the
    river :math:`(x,y)`, and our cow :math:`C`, and also assuming a Euclidean
    distance, we can come up with the total distance the milkmaid needs to walk:

    .. math::

        f(x,y) = \sqrt{(P_x - x)^2 + (P_y - y)^2} + \sqrt{(C_x - x)^2 + (C_y - y)^2}

    From here, you can use the same method as above to solve for :math:`x` and
    :math:`y` with the constraint for the river as :math:`g(x,y)=0`.


.. admonition:: Example 3: Maximum likelihood estimate (MLE) for a multinomial distribution.

    **Problem**: Suppose we have observed :math:`n` rolls of a six-sided die.
    What is the MLE estimate for the probability of each side of the die
    (:math:`p_1, \ldots, p_6`)?

    Recall the log-likelihood of a `multinomial distribution <https://en.wikipedia.org/wiki/Multinomial_distribution>`_ with :math:`n` trials and observations :math:`x_1, \ldots, x_6`: 

    .. math::

        f(p_1, \ldots, p_6) &= \log\mathcal{L}(p_1, \ldots, p_6) \\
        &= \log P(X_1 = x_1, \ldots, X_6 = x_6; p_1, \ldots, p_6) \\
        &= \log\big[ \frac{n!}{x_1! \ldots x_6!} p_1^{x_1} \ldots p_6^{x_6} \big] \\
        &= \log n! - \log x_1! - \ldots - \log x_6! + x_1 \log p_1 + \ldots x_6 \log p_6
        \tag{12}

    This defines our :math:`f` function to maximize with six variables
    :math:`p_1, \ldots, p_6`.  Our constraint is that the probabilities must sum to 1:

    .. math::

        g(p_1, \ldots, p_6) = \sum_{k=1}^6 p_k - 1 = 0 \tag{13}

    Computing the partial derivatives:

    .. math::

        \frac{\partial f(p_1, \ldots, p_6)}{\partial p_k} &= \frac{x_k}{p_k} \tag{14} \\
        \frac{\partial g(p_1, \ldots, p_6)}{\partial p_k} &= 1 \tag{15}

    Equations 13, 14, 15 gives us the following system of equations:

    .. math::

        \frac{x_k}{p_k} &= \lambda \text{ for } k=1,\ldots,6 \\
        \sum_{k=1}^6 p_k - 1 &= 0 \tag{16}

    Solving the system of equations in 16 gives us:

    .. math::

        \lambda &= \sum_{k=1}^{6} x_k \\
        p_k &= \frac{x_k}{\lambda} = \frac{x_k}{\sum_{k=1}^{6} x_k} \tag{17}

    Which is exactly what you would expect from the MLE estimate: the
    probability of a side coming up is proportional to the relative number of
    times you have seen it come up.

|h2| Conclusion |h2e|

Lagrange multipliers are one of those fundamental tools in calculus that some
of us never got around to learning.  This was really frustrating for me when
I was trying to work through some of the math that comes up in probability
texts such the problem above.  Like most things, it is much more mysterious
when you just have a passing reference to it in a textbook versus actually
taking the time to understand the intuition behind it.  I hope this tutorial
has helped some of you along with your understanding.

|h2| Further Reading |h2e|

* Wikipedia: `Lagrange multiplier <https://en.wikipedia.org/wiki/Lagrange_multiplier>`_,
  `Gradient <https://en.wikipedia.org/wiki/Gradient>`_
* `An Introduction to Lagrange Multipliers <http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html>`_, Steuard Jensen
* `Lagrange Multipliers <https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/constrained-optimization/a/lagrange-multipliers-single-constraint>`_, Kahn Academy

|br|

.. [1] This post draws heavily on a great tutorial by Steuard Jensen: `An Introduction to Lagrange Multipliers <http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html>`_.  I highly encourage you to check it out.
