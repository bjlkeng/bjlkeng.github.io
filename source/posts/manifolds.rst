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


|h2| Manifolds |h2e|

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

*(Note: You should check out [1] and [2], which are great YouTube playlists
for understanding these topics.  [2] especially is just as good (or better)
than a lecture at a university.)*


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
    This means that for every point :math:`p` in :math:`M` there is an open
    neighbourhood :math:`U` of :math:`p` and a homeomorphism :math:`\varphi: U \rightarrow V`
    which maps the set :math:`U` onto an open set :math:`V \subset \mathbb{R}^n`.
    Additionally:

    * The mapping :math:`\varphi: U \rightarrow V` is called a **chart** or **coordinate system**.  
    * The set :math:`U` is the **domain** or **local coordinate neighbourhood** of the chart.  
    * The image of the point :math:`p \in U`, denoted by :math:`\varphi(p) \in \mathbb{R}^n`,
      is called the **coordinates** of :math:`p` in the chart.  
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
:math:`p` in our domain, and map it into the lower dimensional Euclidean space,
the mapped point is called the *coordinate* of :math:`p` in our chart.
Finally, if we have a bunch of charts whose domains exactly spans the entire
manifold, then this is called an *atlas*.

The best analogy for all of this is really just geography.  I've never really
studied geography beyond grade school but I'm guessing you have similar terminology
such as charts, coordinate systems, and atlases.  The ideas are, on the
surface, similar.  However, I'd probably still stick with Figure 4, which
is much more accurate.

.. admonition:: Manifolds: All About Mapping

  Wrapping your head around manifolds can be sometimes be hard because of all
  the symbols.  The key thing to remember is that **manifolds it's all about mappings**.
  Mapping from the manifold to a local coordinate system in Euclidean space
  using a chart; mapping from one local coordinate system to another
  coordinate system; and later on we'll also see mapping a curve or function on
  a manifold to a local coordinate too.  Sometime we'll do "hop" once (e.g. manifold
  to local coordanates), or multiple "hops" (parameter of a curve to location on
  a manifold to local coordanates).  And since most of our mappings are 1-1
  we can "hop" back and forth as we please to get the mapping we want.
  So make sure you are comfortable
  with how to do these "hops" which are nothing more than simple `functions
  compositions <https://en.wikipedia.org/wiki/Function_composition>`__.


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

  Standard Euclidean space in :math:`\mathbb{R}^n` is, of course, a manifold
  itself. It requires a single chart that it just the identity function,
  which also makes up its Atlas.  We'll see below that many of concepts
  we've been learned in Euclidean space have analogues when discussing
  manifolds.


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

  Using the "north" pole point, for any other given point :math:`p=(x,y)` on
  the circle, we can find where it intersects the x-axis via similar triangles
  (the radius of the circle is 1, :math:`\frac{\text{adjacent}}{\text{opposite}}`):

  .. math::

    u_1 := \varphi_1(p) = \frac{\varphi_1(p)}{1} = \frac{x_p}{1 - y_p} \tag{2}
    
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
 
.. admonition:: Example 3: Stereographic Projections for :math:`S^n`

  As you might have guessed, we can perform the same 
  `stereographic projection <https://en.wikipedia.org/wiki/Stereographic_projection>`__
  for :math:`S^2` as well.  Figure 6 shows a visualization (nevermind the
  different notation, I used a drawing from Wikipedia instead of trying to make my own :p).

  .. figure:: /images/manifold_sphere.png
    :height: 350px
    :alt: Spherical Manifold
    :align: center
  
    Figure 6: A construction of charts on a 2D sphere (source: Wikipedia).

  In a similar way, we can pick a point, draw a line that intersects any other point on
  the sphere, and project it out to the :math:`z=0` (2D) plane.  This chart can cover
  every point except the starting point.  Using two charts each with a point
  (e.g. "north" and "south" pole), we can create an Atlas that covers every point
  on the sphere.

  In general an `n-dimensional sphere <https://en.wikipedia.org/wiki/N-sphere>`__ 
  is a manifold of :math:`n` dimensions and is given the name :math:`S^n`.
  So a circle is a 1-dimensional sphere, a "normal" sphere is a 2-dimensional sphere,
  and a n-dimensional sphere can be embedded in (n+1)-dimensional Euclidean space
  where each point is equidistant to the origin.  

  This projection can generalized for :math:`S^n` using the same method: 
  
  1. Pick an arbitrary focal point on the sphere (not on the hyperplane you are
     projecting to) e.g. the "north" pole :math:`p_N = (0, \ldots, 0,1)`.
  2. Project a line from the focal point to any other point on the hypersphere.
  3. Pick a plane that intersects at the "equator" relative to the focal point.
     e.g. For the "north" pole focal point, the plane given by the first
     :math:`n` coordinates (remember the :math:`S^n` has n+1 coordinates because
     it's embedded in :math:`\mathbb{R}^{n+1}`.

  From this, we can derive similar formulas (using the same similar triangle
  argument) as the previous example.  Using the north pole, as an example, to any
  point :math:`p =({\bf x}, z) \in S^n` on the hypersphere, and the hyperplane
  given by :math:`z=0`, we can get the projected point :math:`({\bf u_N}, 0)`
  using the following equations:

  .. math::

    {\bf u_N} := \varphi_N(p) = \frac{\bf x}{1 - z} \\
    {\bf x} = \frac{2{\bf u_N}}{|{\bf u_N}|^2 + 1} \\
    z = \frac{|{\bf u_N}|^2 - 1}{|{\bf u_N}|^2 + 1} \\
    \tag{6}

  for vectors :math:`{\bf u_N, x} \in \mathbb{R}^{n}`.  The symmetric equations
  can also be performed for the "south" pole.


|h2| Tangent Spaces |h2e|

To actually calculate things like distance on a manifold, we have to 
introduce a few concepts.  The first is a **tangent space** :math:`T_x M`
of a manifold :math:`M` at a point :math:`{\bf x}`.  It's pretty much exactly
as it sounds: imagine you are walking along a curve on a smooth manifold,
as you pass through the point :math:`{\bf x}` you implicitly have velocity
(magnitude and direction) that is tangent to the manifold or another name
is a  **tangent vector**.  The tangent vectors made in this way from each
possible curve passing through point :math:`{\bf x}` make up the tangent space at
:math:`x`.  In two dimensions, this would be a plan.  Figure 7 shows a
visualization of this on a manifold.

.. figure:: /images/tangent_space_vector.png
  :height: 250px
  :alt: Tangent Vector
  :align: center

  Figure 7: A tangent space :math:`T_x M` for manifold :math:`M` with tangent
  vector :math:`{\bf v} \in T_x M`, along a curve travelling through :math:`x \in M`
  (source: Wikipedia).

I should note that Figure 7 is a bit misleading because the tangent space/vector
doesn't necessarily look literally like a plane tangent to the manifold. 
It *can* however look like this when it is embedded in a higher dimension space
like it is here for visualization purposes 
(e.g. 2D manifold as a surface shown in 3D with a plane tangent to the surface
representing the "tangent space").  Manifolds don't need to even be embedded
in a higher dimensional space (recall that they are defined just as special
sets with a mapping to Euclidean space) so we should be careful with some of
these visualizations.  However, it's always good to have an intuition.  Let's
try to formalize this idea in two steps: the first a bit more intuitive, the
second a deeper look to allow us to perform more operations.

|h3| Tangent Spaces as the Velocity of Curves |h3e|

Suppose we have our good old smooth manifold :math:`M` and a point on that
curve :math:`p \in M`.  For a given coordinate chart 
:math:`\varphi: U \rightarrow \mathbb{R}^n` where :math:`U` is an open subset of 
:math:`M` containing :math:`p`.  So far so good, this is just repeating what we
had in Figure 4.

Now let's define a smooth parametric curve :math:`\gamma: t \rightarrow M` that
maps a parameter :math:`t \in [a,b]` to :math:`M` that passes through
:math:`p`.  Now we want to imagine we're walking along this curve *in the local
coordinates* i.e. after applying our chart (this is where Figure 7 might be
misleading), this will give us: :math:`\varphi \circ \gamma: t \rightarrow
\mathbb{R}^n` (from :math:`t` to :math:`M` to :math:`\mathbb{R}^n` in the local
coordinates).

Let's label our local coordinates as :math:`{\bf u} = \varphi \circ \gamma(t)`,
which is nothing more than a vector valued function of a single parameter which
can be interpreted as your "position" vector on the manifold (in local
coordinates) as a function of time (:math:`t`).  Thus, the velocity is just the
instantaneous rate of change of our position vector with respect to time.  So
at time :math:`t=t_0` when we're at the point :math:`p`, we have:

.. math::

    \text{"velocity" at } p = \frac{d \varphi \circ \gamma(t)}{dt}\Big|_{t=t_0}
                          = \Big[\frac{dx^1(t)}{dt}, \ldots, \frac{dx^n(t)}{dt}\Big]\Big|_{t=t_0} \tag{7}

where :math:`x^i(t)` is the :math:`i^{th}` component of our curve in local coordinates (not an exponent).
In this case the `tangent vector <https://en.wikipedia.org/wiki/Tangent_vector>`__ :math:`\bf v` 
is nothing more than the "velocity" at :math:`p`.
If we then take every possible velocity at :math:`p` (by specifying different
parametric curves) then these velocity vectors make a `tangent space
<https://en.wikipedia.org/wiki/Tangent_space>`__, denoted by :math:`T_pM`
(careful, our point on the manifold is now :math:`p` and the local coordinate
is :math:`x`).
Now that we have our tangent (vector) space represented in
:math:`\mathbb{R}^n`, we can perform our usual Euclidean vector-space
operations.  (We'll get to it in the next section).

|h3| Basis of the Tangent Space |h3e|

Tangent vectors as velocities only tells half the story though because we have
a tangent vector specified in a local coordinate system but what is its basis?
Recall a `vector <https://en.wikipedia.org/wiki/Coordinate_vector>`__ has its
coordinates (an ordered list of scalars) that correspond to particular basis
vectors.  This is important because we want to be able to do analysis on the
manifold *between* points, not just at a single point.  So understanding how
the tangent spaces between different points (and potentially charts) on a
manifold is important.

To understand how to construct the tangent space basis, let's first define
an arbitrary function :math:`f: M \rightarrow \mathbb{R}` and assume we still
have our good old smooth parametric curve :math:`\gamma: t \rightarrow M`.
Now we want to look at a new definition of "velocity" relative to this test
function: :math:`\frac{df \circ \gamma(t)}{dt}\Big|_{t=t_0}` at our point on the manifold
:math:`p`.  Basically the rate of change of our function as we walk along this
curve.

However, we can do a "trick" by introducing a chart (:math:`\varphi`) and its
inverse (:math:`\varphi^{-1}`) into this measure of "velocity":

.. math::

    \frac{df \circ \gamma(t)}{dt}\Big|_{t=t_0} 
    &= \frac{d(f \circ \varphi^{-1} \circ \varphi \circ \gamma)(t)}{dt}\Big|_{t=t_0}  \\
    &= \frac{d((f \circ \varphi^{-1}) \circ (\varphi \circ \gamma))(t)}{dt}\Big|_{t=t_0}  \\
    &= \sum_i \frac{\partial (f \circ \varphi^{-1})(x)}{\partial x_i}\Big|_{x=\varphi \circ \gamma(t_0)}
       \frac{d(\varphi \circ \gamma)^i(t)}{dt}\Big|_{t=t_0} && \text{chain rule} \\
    &= \sum_i \frac{\partial (f \circ \varphi^{-1})(x)}{\partial x_i}\Big|_{x=\varphi(p)}
       \frac{d(\varphi \circ \gamma)^i(t)}{dt}\Big|_{t=t_0} && \text{since }\varphi(p) = \gamma(t_0) \\
    &= \sum_i (\text{basis for component }i)(\text{"velocity" of component i wrt to } \varphi) \\
    \tag{8}

Note the introduction of partial derivatives and summations in the third line,
which is just an application of the multi-variable calculus chain rule.
We can see that by introducing this test function and doing our little trick we
get the same velocity as Equation 7 but with its corresponding basis vectors.

Okay the next part is going to be a bit strange but bear with me.  We're going
to take the basis and re-write like so:

.. math::

    \Big(\frac{\partial}{\partial x^i}\Big)_p (f) := \frac{\partial (f \circ \varphi^{-1})(\varphi(p))}{\partial x_i} \\
    \tag{9}

which simply defines some new notation for the basis.  Importantly the LHS now
has no mention of :math:`\varphi` anymore, but why?  Well there is a convention
that :math:`\varphi` is implicitly specified by :math:`x^i`.  So if you have
some other chart, say, :math:`\vartheta`, then you label its local coordinates
with :math:`y^i`.  But it's important to remember that when we're using this
notation, implicitly there is a chart behind it.

Okay, so now that we've cleared up that, there's another thing we need to look at:
what is :math:`f`?  We know it's some test function that we used, but it was 
arbitrary.  And in fact, it's so arbitrary we're going to get rid of it!  So
we're just going to define the basis in terms of the *operator* that acts of
:math:`f` and not the actual resultant vector!  So every tangent vector
:math:`v \in T_pM`, we have:

.. math::

    {\bf v} &= \sum_{i=1}^n v(x^i) \cdot \Big(\frac{\partial}{\partial x^i}\Big)_p \\
      &= \sum_{i=1}^n \frac{d(\varphi \circ \gamma)^i(t)}{dt}\Big|_{t=t_0}  \cdot
        \Big(\frac{\partial}{\partial x^i}\Big)_p \\
        \tag{10}

It turns out the basis is actually a set of *differential operators* (not the
actual vectors on the test function :math:`f`), which make up a `vector space
<https://en.wikipedia.org/wiki/Vector_space#Definition>`__ (with respect to
chart :math:`\varphi`)!  A bit mind bending if you're not used to these
abstract definitions (vector spaces don't have anything strictly to do with
Euclidean vectors, it's just that we first learn about them through Euclidean
vectors)

|h3| Change of Basis for Tangent Vectors |h3e|

Now that we have a basis for our tangent vectors, we want to understand how to
change basis between them.  Let's just setup/recap a big of notation first.
Let's define two charts for our :math:`d`-dimensional manifold :math:`M`:

.. math::

    \varphi(p) = (x^1(p), \ldots, x^d(p)) \\
    \vartheta(p) = (y^1(p), \ldots, y^d(p)) \\
    \tag{11}

where :math:`x^i(p)` and :math:`y^i(p)` are coordinate functions to find the
specific index of the local coordinates from a point on our manifold :math:`p
\in M`.  Assume that :math:`p` is in the overlap of the domains in the two
charts.  Now we want to look at how we can convert from a tangent space in one
chart to another.

(We're going to switch to a more convenient 
`partial derivative notation <https://en.wikipedia.org/wiki/Partial_derivative>`__
here: :math:`\partial_x f := \frac{\partial f}{\partial x}`, which is just a
bit more concise.)

So starting from our summation in Equation 10 (and using `Einstein summation
notation <https://en.wikipedia.org/wiki/Einstein_notation>`__, see also previous
my post on `Tensors <link://slug/tensors-tensors-tensors>`__)
acting on our test function :math:`f`:

.. math::

    {\bf v} f &= v(x^i) \cdot \Big(\frac{\partial}{\partial x^i}\Big)_p f \\
      &= v(x^i) \cdot \partial_{x^i} (f \circ \varphi^{-1})(\varphi(p)) && \text{by definition} \\ 
      &= v(x^i) \cdot \partial_{x^i} (f \circ \vartheta^{-1} \circ \vartheta \circ \varphi^{-1})(\varphi(p)) && \text{introduce } \vartheta \text{ with identity trick}\\ 
      &= v(x^i) \cdot \partial_{x^i} ((f \circ \vartheta^{-1}) \circ (\vartheta \circ \varphi^{-1}))(\varphi(p)) \\ 
      &= v(x^i) \cdot 
        \partial_{x^i} (\vartheta \circ \varphi^{-1})^j(\varphi(p))
        \cdot
        \partial_{y^j} (f \circ \vartheta^{-1})(\vartheta \circ \varphi^{-1}(\varphi(p)))
        && \text{chain rule} \\
      &= v(x^i) \cdot 
        \partial_{x^i} (\vartheta \circ \varphi^{-1})^j(\varphi(p))
        \cdot
        \partial_{y^j} (f \circ \vartheta^{-1})(\vartheta(p))
        && \text{simplifying} \\
      &= v(x^i) \cdot 
        \partial_{x^i} (\vartheta \circ \varphi^{-1})^j(\varphi(p))
        \cdot
        \Big(\frac{\partial}{\partial y^i}\Big)_p f
        && \text{by definition} \\
      &= v(x^i) \cdot 
        \frac{\partial y^j}{\partial x^i}\big|_{x=\varphi(p)}
        \cdot
        \Big(\frac{\partial}{\partial y^i}\Big)_p f
        && \text{since }y^j(x) = y^j(\varphi(p)) \\
      &= v(y^i) \cdot \Big(\frac{\partial}{\partial y^j}\Big)_p f \\
    \tag{12}
    
After some wrangling with the notation, we can see the change of basis
is basically just an application of the chain rule.  If you squint hard
enough, you'll see the change of basis matrix is simply the Jacobian
:math:`J` of :math:`\vartheta(x) = (y^1(x), \ldots, y^d(x))` written with
respect to the original chart coordinates :math:`x^i` (instead of the manifiold
point :math:`p`).

So after all that manipulation, let's take a look at an example on our
sphere to make things a bit more concrete.

.. admonition:: Example 4: Tangent Vectors on a Sphere

  On our unit sphere, let's try to find the tangent vector at point
  :math:`p = (x,y,z) = (1, 0, 0)` on the equator of the sphere.  We'll
  use our "north" pole chart and a curve that is orbiting the equator.
  Let's setup the problem.

  First, following Equation 6, our chart :math:`\varphi` (and its inverse) look
  like (I'm not going to use the superscript notation here for the local
  coordinates but you should know it's pretty common):

  .. math::

    u_1(x, y, z) &= \frac{x}{1-z} \\
    u_2(x, y, z) &= \frac{y}{1-z} \\
    x &= \frac{2u_1}{\sqrt{x^2 + y^2} + 1} \\
    y &= \frac{2u_2}{\sqrt{x^2 + y^2} + 1} \\
    z &= \frac{\sqrt{x^2 + y^2}- 1}{\sqrt{x^2 + y^2} + 1} \\
    \tag{13}

  Next, let's define our curve: :math:`\gamma(t) = (\cos\pi t, \sin\pi t, 0), t\in[-1, 1]`,
  which we can see is on the equator, outlining a circle on the :math:`z=0`
  plane.  We can also see at :math:`t_0=0`, :math:`\gamma(t=0) = (1, 0, 0) = p`.

  To find the coordinates in our tangent space, we use Equation 7:

  .. math::
  
      \frac{d \varphi \circ \gamma(t)}{dt}\Big|_{t=t_0}
      &= \Big[
      \frac{d u_1(\cos\pi t, \sin\pi t, 0)}{dt},  
      \frac{d u_2(\cos\pi t, \sin\pi t, 0)}{dt}  
      \Big]\Big|_{t=t_0}  \\
      &= \Big[
      \frac{d \cos\pi t}{dt},
      \frac{d \sin\pi t}{dt}
      \Big]\Big|_{t=t_0}  \\      
      \tag{14}
      &= \Big[ -\sin\pi t, \cos\pi t \Big]\Big|_{t=t_0}  \\      
      &= (0, 1)
      \tag{14}

  Combining with out differential operators as our basis, our tangent vector
  becomes:

  .. math::

    v = 0 \cdot \big(\frac{\partial}{\partial u_1} \big)_p +
        1 \cdot \big(\frac{\partial}{\partial u_2} \big)_p \tag{15}

  keeping in mind that the basis is actually in terms of the chart :math:`\varphi`
  (implied by the variable :math:`u_i`).


|h2| Riemannian Manifolds |h2e|

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

* Previous posts: `Tensors, Tensors, Tensors <link://slug/tensors-tensors-tensors>`__
* Wikipedia: `Manifold <https://en.wikipedia.org/wiki/Manifold>`__,
  `Metric Tensor <https://en.wikipedia.org/wiki/Metric_tensor>`__,
  `Metric Space <https://en.wikipedia.org/wiki/Metric_space>`__,
* `Differentiable manifolds and smooth maps <http://www.maths.manchester.ac.uk/~tv/Teaching/Differentiable%20Manifolds/2010-2011/1-manifolds.pdf>`__, Theodore Voronov.
* [1] `"Manifolds (playlist)" <https://www.youtube.com/playlist?list=PLeFwDGOexoe8cjplxwQFMvGLSxbOTUyLv>`__, Robert Davie (YouTube)
* [2] `"What is a Manifold?" <https://www.youtube.com/playlist?list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo>`__, XylyXylyX (YouTube)
