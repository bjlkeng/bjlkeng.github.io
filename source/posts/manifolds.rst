.. title: Manifolds: A Gentle Introduction
.. slug: manifolds
.. date: 2018-03-13 16:24:57 UTC-05:00
.. tags: manifolds, metric tensor, mathjax
.. category:
.. link:
.. description: A quick introduction to manifolds.
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

.. |hr| raw:: html

   <hr>

Following up on the math-y stuff from my `last post <link://slug/tensors-tensors-tensors>`__, 
I'm going to be taking a look at another concept that pops up in ML: manifolds.
It's probably most well-known use in ML is the 
`manifold hypothesis <https://www.quora.com/What-is-the-Manifold-Hypothesis-in-Deep-Learning>`__.
Manifolds belong to the branch of mathematics called topology and differential
geometry.  I'll be focusing more on the study of manifolds from the latter
category, which fortunately is a bit less abstract and more intuitive than the
former.  As usual, I'll go through some intuition, definitions, and examples to
help clarify the ideas without going too much into formalities.  I hope you
mani-like it!


.. TEASER_END


|h2| Manifold Motivation |h2e|

The first place most ML people hear about this term is in the 
`manifold hypothesis <https://www.quora.com/What-is-the-Manifold-Hypothesis-in-Deep-Learning>`__:

    The manifold hypothesis is that real-world high dimensional data (such as
    images) lie on low-dimensional manifolds embedded in the high-dimensional
    space.

The main idea here is that even though our real-world data is high-dimensional,
there is actually some lower-dimensional representation.  For example, all "cat
images" might lie on a lower-dimensional manifold compared to say their
original 256x256x3 image dimensions.  This makes sense because we are
empirically able to the learn these things in a capacity limited neural
network.  Otherwise, otherwise learning an arbitrary 256x256x3 function would
be intractable.

Okay, that's all well and good, but that still answer the question: *what is a manifold?*
The abstract definition from topology is well... abstract.  So I won't go into
all the technical details (also because I'm not very qualified to do so), but
we'll see that the differential and Riemannian manifolds are surprisingly
intuitive (in low dimensions at least) once you get the hang (or should I say
twist) of things.


|h3| Circles and Spheres as Manifolds |h3e|

A **manifold** is topological space that "locally" resembles Euclidean space.
This obviously doesn't mean much unless you've studied topology.  
An intuitive (but not exactly correct) way to think about it is taking
a geometric object from :math:`\mathbb{R}^k` and trying to "fit" it into
:math:`\mathbb{R}^n, n>k`.  Let's take a first example, a line segment, which
is obviously one dimensional. 

One way to embed a line in two dimensions is to "wrap" it around into a circle,
shown in Figure 1.  Each arc of the circle locally looks closer to a line
segment, and if you take an infinitesimal arc, it will "locally" resemble a
one dimensional line segment.  

.. figure:: /images/manifold_circle.png
  :height: 250px
  :alt: A Circle is a Manifold
  :align: center

  Figure 1: A circle is a manifold in two dimensions where each arc of the
  circle is locally resembles a line (source: Wikipedia).

Of course, there is a much more 
`precise definition <https://en.wikipedia.org/wiki/Topological_manifold#Formal_definition>`__ 
from topology in which a manifold is defined as a set that is 
`homeomorphic <https://en.wikipedia.org/wiki/Homeomorphism>`__ 
to a Euclidean space.  A homeomorphism is a special kind of continuous one-to-one 
mapping that preserves topological properties.  The definition is quite
abstract because the definition says a (topological) manifold is just a special
kind of set without any explicit reference of how it can be viewed as a
geometric object.  We'll take a closer look at this a bit below, but for now,
let's focus on the big idea.

Actually any "closed" loop in one dimension is a manifold because you can
imagine "wrapping" it around into the right shape.  Another way to think about
it (from the formal definition) is that from a line (segment), you can find a
continuous one-to-one mapping to a closed loop.  An interesting point is that
figure "8" is not a manifold because the crossing point does not locally
resemble a line segment.

These closed loop manifolds are the easiest 1D manifolds to think about but
there are other weird cases too shown in Figure 2.  As you can see, we can have
a variety of different shapes.  The big idea is that we can also have "open
ended" curves that extend out to infinity, which are natural mappings to
a one dimensional line.

.. figure:: /images/manifold_1d_other.png
  :height: 250px
  :alt: Other 1D Manifolds
  :align: center

  Figure 2: Circles, parabolas, hyperbolas and 
  `cubic curves <https://en.wikipedia.org/wiki/Cubic_curve>`__ 
  are all 1D Manifolds.  Note: the four different colours are all on separate
  axes and extend out to infinity if it has an open end (source: Wikipedia).

Let's now move onto 2D manifolds. The simplest one is a sphere.  You can
imagine each infinitesimal patch of the sphere locally resembles a 2D Euclidean
plane.  Similarly, any 2D surface (including a plane) that doesn't
self-intersect is also a 2D manifold.  Figure 3 shows some examples.

.. figure:: /images/manifold_2d.gif
  :height: 350px
  :alt: 1D Manifolds
  :align: center

  Figure 3: Non-intersecting closed surfaces in :math:`\mathbb{R}^3` are 
  examples of 2D manifolds such as a sphere, torus, double torus, cross
  surfaces and Klein bottle (source: Wolfram).

For these examples, you can imagine that each point on these manifolds
locally resembles a 2D plane.  This best analogy is Earth.  We know that the
Earth is round but when we stand in a field it looks flat.  We can of course
have higher dimension manifolds embedded in even larger dimension Euclidean
spaces but you can't really visualize them.  Abstract math is rarely easy to
visualize in higher dimension.

Hopefully after seeing all these examples, you've developed some intuition
around manifolds.  In the next section, we'll head back to the math with some
differential geometry.

|h3| A (slighly) More Formal Look at Manifolds |h3e|

Now that we have some intuition, let's a first look at the formal definition of
`(topological) manifolds <https://en.wikipedia.org/wiki/Topological_manifold#Formal_definition>`__, 
which I took from [1]:


    An n-dimensional **topological manifold** :math:`M` is a topological Hausdorff
    space with a countable base with is locally homeomorphic to :math:`\mathbb{R}^n`.
    This means that for every point :math:`P` in :math:`M` there is an open
    neighbourhood :math:`U` of :math:`P` and a homeomorphism :math:`\varphi: U \rightarrow V`
    which maps the set :math:`U` onto an open set :math:`V \subset \mathbb{R}^n`.
    Additionally:

    * The mapping :math:`\varphi: U \rightarrow V` is called a **chart** or **coordinate system**.  
    * The set :math:`U` is the **domain** or **local coordinate neighbourhood** of the chart.  
    * The image of the point :math:`P \in U`, denoted by :math:`\varphi(P) \in \mathbb{R}^n`,
      is called the **coordinates** of :math:`P` in the chart.  
    * A set of charts, :math:`\{\varphi_\alpha | \alpha \in \mathbb{N}\}`, with domains :math:`U_\alpha`
      is called the **atlas** of M, if :math:`\bigcup\limits_{\alpha \in \mathbb{N}} U_\alpha = M`.
   
This definition is hard to understand especially because a Hausdorff space is
never defined.  That's not too important because we're not going to go into the
topological formalities, the most important parts are the new terminology,
which thankfully have an intuitive interpretation.  Let's take a look at Figure 4,
which should clear up some of the ideas.

.. figure:: /images/coordinate_chart_manifold.png
  :height: 250px
  :alt: Charts on a Manifold 
  :align: center

  Figure 4: Two intersecting patches (green and purple with cyan/teal as the
  intersection) on a manifold with different charts (continuous 1-1 mappings)
  to 2D Euclidean space.  Notice that the intersection of the patches have a
  smooth 1-1 mapping in 2D Euclidean space, making it a differential manifold
  (source: Wikipedia).


First of all our manifold in this case is :math:`X`, which we can imagine is
embedded in some high dimension :math:`n+k`.
We have two different "patches" or *domains* (or *local coordinate neighbourhoods*)
defined by :math:`U_\alpha` (green) and :math:`U_\beta` (purple) in :math:`X`.
Since it's a manifold, we know that each point locally has a
mapping to a lower dimensional Euclidean space (say :math:`\mathbb{R}^n`) via
:math:`\varphi`, our *chart* or *coordinate system*.  If we take a point
:math:`P` in our domain, and map it into the lower dimensional Euclidean space,
the mapped point is called the *coordinate* of :math:`P` in our chart.
Finally, if we have a bunch of charts whose domains exactly spans the entire
manifold, then this is called an *atlas*.

The best analogy for all of this is really just geography.  I've never really
studied geography beyond grade school but I'm guessing you have similar terminology
such as charts, coordinate systems, and atlases.  The ideas are, on the
surface, similar.  However, I'd probably still stick with Figure 4, which
is much more accurate.

Figure 4 also has another mapping between the intersecting parts of
:math:`U_\alpha` and :math:`U_\beta` in their respective chart coordinates
called a **transition map**, given by 
:math:`\varphi_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1}` and 
:math:`\phi_{\beta\alpha}=\varphi_\alpha \circ \varphi_\beta^{-1}` 
(their domain is restricted to either :math:`\varphi_\alpha(U_\alpha \cap U_\beta)`
or :math:`\varphi_\beta(U_\alpha \cap U_\beta)`, respectively).

These transition functions are important because depending on their
differentiability, they define a new class of 
`differentiable manifolds <https://en.wikipedia.org/wiki/Differentiable_manifold>`__
(denoted by :math:`C^k` if they are k-times continuously differentiable).
The most important one for our conversation being transition maps that are
infinitely differentiable, which we call 
`smooth manifolds <https://en.wikipedia.org/wiki/Differentiable_manifold#Definition>`__.

The motivation here is that once we have smooth manifolds, we can do a bunch of
nice things like calculus.  Remember, once we have smooth mappings to lower
dimensional Euclidean space, things are a lot easier to analyze.  Performing
analysis on a manifold embedded in a high dimensional space could be a major
pain in the butt, but analysis in a lower-dimensional Euclidean space is easy
(relatively)!

.. admonition:: Example 1: Euclidean Space is a Manifold

  This is an example


.. admonition:: Example 2: A 1D Manifold with Multiple Charts

  Let's take pretty much the simplest example we can think of: a circle.

  If we use `polar coordinates <https://en.wikipedia.org/wiki/Polar_coordinate_system#Conventions>`__,
  the unit circle can be parameterized with :math:`r=1` and :math:`\theta`.

  The unit circle is a 1D manifold :math:`M`, so it should be able to map to
  :math:`\mathbb{R}`.  We might be tempted to just have a simple chart mapping
  such as :math:`\varphi(r, \theta) = \theta` but because :math:`\theta` is a
  multi-valued we need to restrict the domain.  Further, we'll need more than
  one chart mapping because a chart can only work on an open set 
  (the analogue to an open interval, i.e. we can't use :math:`[0, 2\pi)`).
  
  We can create four charts (or mappings) as in Figure 1, that have the form
  :math:`M \rightarrow \mathbb{R}`:

  .. math::

    \varphi_1(r, \theta) &= \theta  && \theta \in (-\frac{\pi}{3}, \frac{\pi}{3}) \\
    \varphi_2(r, \theta) &= \theta  && \theta \in (\frac{\pi}{6}, \frac{5\pi}{6}) \\
    \varphi_3(r, \theta) &= \theta  && \theta \in (\frac{2\pi}{3}, \frac{4\pi}{3}) \\
    \varphi_4(r, \theta) &= \theta  && \theta \in (\frac{7\pi}{6}, \frac{11\pi}{6}] \\
    \tag{1}

  Notice that there is overlap in :math:`\theta` between the charts where each
  one has an open set (i.e. the domain) on the original circle.
  :math:`{\varphi_1, \varphi_2, \varphi_3, \varphi_4}` together form an atlas
  for :math:`M` because their domains span the entirety of the manifold.

  |hr|

  We can also find other charts to map the unit circle.  Let's take a look at
  another construction using standard Euclidean coordinates and a 
  `stereographic projection <https://en.wikipedia.org/wiki/Stereographic_projection>`__.
  Figure 5 shows a picture of this construction.

  .. figure:: /images/circle_manifold_projection.png
    :height: 350px
    :alt: Circle Manifold with Charts
    :align: center
  
    Figure 5: A construction of charts on a 1D circle manifold.

  We can define two charts by taking either the "north" or "south" pole of the
  circle, finding any *other* point on the circle and projecting the line
  segment onto onto the x-axis.  This provides the mapping from a point on the
  manifold to :math:`\mathbb{R}^1`.  The "north" pole point is visualized in blue,
  while the "south" pole point is visualized in burgundy.  Note: the local
  coordinates for the charts are *different*.  The same two points on the circle
  for the two charts, do not map to the same point in :math:`\mathbb{R}^1`.

  Using the "north" pole point, for any other given point :math:`P=(x,y)` on
  the circle, we can find where it intersects the x-axis via similar triangles
  (the radius of the circle is 1, :math:`\frac{\text{adjacent}}{\text{opposite}}`):

  .. math::

    u_1 := \varphi_1(P) = \frac{\varphi_1(P)}{1} = \frac{x_p}{1 - y_p} \tag{2}
    
  This defines a mapping for every point on the circle except the "north" pole.
  Similarly, we can define the same mapping for the "south" pole for any 
  point on the circle :math:`Q` (except the "south" pole):

  .. math::
  
     u_2 := \varphi_2(Q) = \frac{\varphi_2(Q)}{1} = \frac{x_q}{1 + y_q} \tag{3}
  
  Together, :math:`{\varphi_1, \varphi_2}` make up an atlas for :math:`M`.
  Since charts are 1-1, we can find the inverse mapping between the manifold
  and local coordinates as well (using the fact that :math:`x^2 + y^2=1`):

  .. math::

     x_p &= \frac{2u_1}{u_1^2+1}, &y_p = \frac{u_1^2-1}{u_1^2+1} \\
     x_q &= \frac{2u_2}{u_2^2+1}, &y_q = \frac{1-u_2^2}{u_2^2+1} \\
     \tag{4}

  Finally, we can find the transition map :math:`\varphi_{\alpha\beta}` as:

  .. math::

    u_2 &= \varphi_{\alpha\beta}(u_1) \\
    &= \varphi_2 \circ \varphi_1^{-1}(u_1) \\
    &= \varphi_2(\varphi_1^{-1}(u_1)) \\
    &= \varphi_2\big((\frac{2u_1}{u_1^2+1}, \frac{u_1^2-1}{u_1^2+1})\big) \\
    &= \frac{\frac{2u_1}{u_1^2+1}}{1 + \frac{u_1^2-1}{u_1^2+1}} \\
    &= \frac{1}{u_1} \\
    \tag{5}

  which is only defined for the points in the intersection (i.e. all points
  on the circle except the "north" and "south" pole).
 
.. admonition:: Example 3: Stereographic Projections for a Sphere

  This is an example

|h3| Tangent Spaces |h3e|



|h3| Riemannian Manifolds |h3e|

To actually calculate things like distance on a manifold, we have to 
introduce a few concepts.  The first is a **tangent space** :math:`T_x M`
of a manifold :math:`M` at a point :math:`x`.  It's pretty much exactly
as it sounds: imagine you are passing through the point :math:`x` on a smooth
manifold, as you pass though you implicitly have a **tangent vector** along the
direction of travel, which can be thought of as your velocity through point :math:`x`.
The tangent vectors made in this way from each possible path passing through
:math:`x` make up the tangent space.  In two dimensions, this would be a plan.
Figure 4 shows an example of this on a sphere.

.. figure:: /images/tangent_space.png
  :height: 250px
  :alt: Tangent Space
  :align: center

  Figure 4: A tangent space at a point on a 2D manifold (a sphere) (source:
  Wikipedia).

In more detail, let's define the curve as for a manifold embedded in
:math:`\mathbb{R}^n` space with start and end points :math:`t \in [a, b]` as:

.. math::

    {\gamma}(t) := [x^1(t), \ldots, x^n(t)] \tag{1}

where :math:`x^i(t)` are single output functions of :math:`t` for component
:math:`i` (not exponents.  In this case the `tangent vector
<https://en.wikipedia.org/wiki/Tangent_vector>`__ :math:`\bf v` at :math:`x` is
just given by the derivative of :math:`\gamma(t)` with respect to :math:`t`:

.. math::

    {\bf v} := \frac{d\gamma(t)}{dt}\Big|_{t=t_x} = \Big[\frac{dx^1(t_x)}{dt}\Big|_{t=t_x}, \ldots, \frac{x^n(t)}{dt}\Big|_{t=t}\Big] \tag{2}
    
where :math:`\gamma(t_x) = x`.  Figure 5 shows another visualization of this
idea with curve :math:`{\bf \gamma}(t)` on the manifold :math:`M`. 

.. figure:: /images/tangent_space_vector.png
  :height: 250px
  :alt: Tangent Vector
  :align: center

  Figure 5: A tangent space :math:`T_x M` for manifold :math:`M` with tangent
  vector :math:`{\bf v} \in T_x M`, along a curve travelling through :math:`x \in M`
  (source: Wikipedia).


You'll be happy to know that tangent vectors are actually *contravariant*
(we didn't waste all that time talking about tensors for nothing)!  We'll
show this fact a bit later.

Back to this smooth manifold we've been talking about, we'll want to define
another tensor, you guess it, the metric tensor!  In particular, the
**Riemannian metric (tensor)** is a family of inner products:

.. math::

    g_p: T_pM \times T_pM \rightarrow \mathbb{R}, p \in M \tag{3}

such that :math:`p \rightarrow g_p(X(p), Y(p))` for any two tangent vectors
(derived from the vector fields :math:`X, Y`) is a smooth function of
:math:`p`.  The implications of this is that even though each adjacent tangent space
can be different (the manifold curves therefore the tangent space changes),
the inner product varies smoothly between adjacent points.  A real, smooth
manifold with a Riemannian metric (tensor) is called a **Riemannian manifold**.


- **Riemannian metric tensor** 
- **Riemannian manifold** 

|h3| Computing Arc Length |h3e|

- Defines arc length, thus distance using differential geometry, inner product
- Explain how to change coordinates 
- Example of computing arc length of a circle with the metric tensor,
  show it in another basis (polar coordinates?)
- Example for unit sphere???

|h2| Metric Space |h2e|

In this section, I just want to quickly review the definition of a metric space
because we'll see that it comes up below.

    A metric space for a set :math:`M` and 
    metric :math:`d: M x M \rightarrow \mathcal{R}` such that for any 
    :math:`x,y,z \in M`, the following holds:

    1. :math:`d(x, y) \geq 0`
    2. :math:`d(x,y)=0 \leftrightarrow x=y`
    3. :math:`d(x,y)=d(y,x)`
    4. :math:`d(x,z)\leq d(x,y) + d(y,z)`

The basic idea is just some space with a distance defined.  The conditions on
your distance function are probably pretty intuitive, although you might
not have explicitly thought about them (the last condition is the triangle
inequality).  Be careful with the definition of "metric", sometimes it's used
for the distance function here, and sometimes it's used for the metric tensor.

Some example of common distance functions:

* :math:`\mathcal{R}^n` with the usual Euclidean distance function
* A `normed vector space <https://en.wikipedia.org/wiki/Normed_vector_space>`__
  using :math:`d(x,y) = ||y-x||` as your metric.  Examples of a norm can be
  the Euclidean distance or Manhattan distance.
* The edit distance between two strings is a metric.

The main thing that I want to emphasize here is that to get a metric space, all
you need is a proper distance function.  We saw in the last section that
we could use the Metric tensor to define an inner product operation, which
allowed us to compute distances.  This idea will come up again below to help
us compute distances on manifolds.



|h2| Conclusion |h2e|



|h2| Further Reading |h2e|

* Wikipedia: `Manifold <https://en.wikipedia.org/wiki/Manifold>`__,
  `Metric Tensor <https://en.wikipedia.org/wiki/Metric_tensor>`__,
  `Metric Space <https://en.wikipedia.org/wiki/Metric_space>`__,
* `Differentiable manifolds and smooth maps <http://www.maths.manchester.ac.uk/~tv/Teaching/Differentiable%20Manifolds/2010-2011/1-manifolds.pdf>`__, Theodore Voronov.
* [1] `"Manifolds (playlist)" <https://www.youtube.com/playlist?list=PLeFwDGOexoe8cjplxwQFMvGLSxbOTUyLv>`__, Robert Davie (YouTube)
* [2] `"What is a Manifold?" <https://www.youtube.com/playlist?list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo>`__, XylyXylyX (YouTube)
