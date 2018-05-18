.. title: Hyperbolic Geometry and Poincaré Embeddings
.. slug: hyperbolic-geometry-and-poincare-embeddings
.. date: 2018-04-20 08:20:18 UTC-04:00
.. tags: manifolds, hyperbolic, geometry, poincaré, embeddings, mathjax
.. category: 
.. link: 
.. description: An introduction to models of hyperbolic geometry and its application to Pointcare embeddings.
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


This post is finally going to get back to some ML related topics.
In fact, the original reason I took that whole math-y detour in the previous
posts was to the more deeply understand this topic, it turns out trying to
under tensor calculus and differential geometry (even to a basic level) takes a
while!  Who knew?  In any case, we're getting back to our regularly scheduled program.

In this post, I'm going to write about a different model of space called
hyperbolic space.  The reason why this abstract math is of interest is because
there has been a surge of research showing its application in various fields,
chief among them is the paper by Facebook [1], which discusses how to utilize
a model of hyperbolic space to represent hierarchical relationships.
As usual, I'll cover some of the math weighting more towards intuition, and
also play around with some implementations to help understand the topic a bit
more.  Don't worry, this time I'll try much harder not going to go down the
rabbit hole of trying to explain all math, rather I'll just stick with a more
intuitive explanation with a sprinkle of math.

(Note: If you're unfamiliar with tensors or manifolds, I suggest getting a quick
overview with my two previous posts: 
`Tensors, Tensors, Tensors <link://slug/tensors-tensors-tensors>`__ and 
`Manifolds: A Gentle Introduction <link://slug/manifolds>`__)

.. TEASER_END


|h2| Curvature |h2e|

To begin this discussion, we have to first understand something about
`curvature <https://en.wikipedia.org/wiki/Curvature>`__.  There are all
kinds of curvature to talk about whether they be on curves, or surfaces (or
hypersurfaces).  With the latter having many different variants.  The basic
idea behind all these different types are that **curvature** is some measure by
which a geometric object deviates from a flat plane, or in the case of a curve,
deviates from a straight line.  

As a side note, there can be **extrinsic curvature** which is defined for
objects embedded in another space (usually Euclidean).  Alternatively, there is
a concept of **intrinsic curvature**, which is defined in terms of lengths of a
curve within a Riemannian manifold, i.e. it is defined independently of any
embedding.  We won't go into all the details but those two terms might come up
when you're reading further.


|h3| Gaussian Curvature |h3e|

To begin, let's start with Gaussian curvature, which is a measure of curvature
for surfaces (2D manifolds).  Let's take a look at Figure 1 which shows three
the three different types of `Gaussian curvature <https://en.wikipedia.org/wiki/Gaussian_curvature>`__.

.. figure:: /images/gaussian_curvature.png
  :height: 170px
  :alt: Examples of Gaussian Curvature
  :align: center

  Figure 1: Examples of the three different types of Gaussian curvature (source: `Science4All <http://www.science4all.org/article/brazuca/>`__).

This gives us a good intuition about what it means to have curvature.  Starting
with the centre diagram (zero curvature), we see that a cylindrical surface has
*flat or zero* in one dimension (along its length) and curved around the other
dimension, resulting in zero curvature.  Moving to the right, the sphere on has
curvature along its to axis in the *same direction*, resulting in a positive
curvature.  And to the left, we see the saddle sheet has curvature along its
axis in *different directions*, resulting in negative curvature.
In fact, the Gaussian curvature is the product of its two 
`principal curvatures <https://en.wikipedia.org/wiki/Principal_curvature>`__, 
which correspond to our intuitive definition of curving along its two different
axes.

Curvature isn't a uniform property of a surface, it's actually defined point-wise.
Figure 2 shows a Torus that has all three different types of Gaussian curvature.
We won't go deeply into the definition of Gaussian curvature because it's a pain but
I think this should probably give some decent intuition on it.

.. figure:: /images/torus_curvature.jpg
  :height: 170px
  :alt: Point-wise curvature on a Torus
  :align: center

  Figure 2: A torus' surface has all three types of curvature.  The outside has positive curvuature (red); the inside has negative curvature (blue); and a ring on the top and bottom of the torus have zero curvature (yellow-orange) (source: `Stackexchange <https://mathematica.stackexchange.com/questions/61409/computing-gaussian-curvature>`__).

Gaussian curvature gives us great intuition on 2D surfaces but what about
higher dimension?  The concept of "axes" gets a bit more muddy, so we need an
alternate way to define things.  One way is to look at the "deviation" of
certain geometric objects such as a triangle.  Figure 3 shows how a triangle on
a surface of differently curved surfaces behave.

.. figure:: /images/Angles-and-Curvature.png
  :height: 170px
  :alt: Deviation of a triangle under the three different types of Gaussian Curvature
  :align: center

  Figure 3: Deviation of a triangle under the three different types of Gaussian Curvature (source: `Science4All <http://www.science4all.org/article/brazuca/>`__).

We can see that in our flat geometry (centre), a triangle's angles add up to
exactly :math:`180^{\circ}`; on positive curvature surfaces (right), it adds up
to more than :math:`180^{\circ}`; on negative curvature surfaces (left), it
adds up to less than :math:`180^{\circ}`.  Measuring how an object's behavior
differs from flat space is (one way) how we're going to generalize curvature to
higher dimensions in the next subsection.



|h3| Parallel Transport, Riemannian Curvature Tensor and Sectional Curvature |h3e|

The first idea we need to get an intuition on is the concept of 
`parallel transport <https://en.wikipedia.org/wiki/Parallel_transport>`__.
The main idea is that we can move tangent vectors along the surface of smooth
manifolds to see if our surface is curved.  

To illustrate, imagine yourself in a tennis court.  First stand in one corner
of the tennis court perpendicularly facing the net with your racket parallel to
the ground pointing forward (the racket represents the tangent vector).
While keeping your racket facing the same direction at all times, walk around the
outside tennis court.  When you get back to your starting position, the racket
is in the same direction.  This is an example of a flat (or zero curvature)
manifold because the vector has not deviated from its original position. 

Now consider another example shown in Figure 4.

.. figure:: /images/parallel_transport.png
  :height: 270px
  :alt: Transporting a tangent vector along the surface of a curved manifold
  :align: center

  Figure 4: Transporting a tangent vector along the surface of a curved
  manifold.  Notice how the direction of the vector changes when traveling from
  A to N to B back to A (source: Wikipedia).

From Figure 4, imagine now that we're on the surface of the earth at point A
facing north with our racket, still parallel to the ground, facing forward i.e.
our tangent vector.  We walk straight all the way up to the north pole at N,
then without changing the direction of our racket, we move towards point B from
the north pole.  Again, without changing our racket direction, we walk back to
point A.  This time though, our racket is pointing in a different direction?!
This is an example of a curved surface, when we parallel transported the vector,
it changed directions.

We can measure this deviation of parallel transport using the 
`Riemannian Curvature Tensor <https://en.wikipedia.org/wiki/Riemann_curvature_tensor>`__.
Starting from a point and moving a vector around a loop, this tensor directly
measures the failure of the vector to point in the initial direction.
For each point on a smooth manifold, it provides a (1, 3)-tensor, which can be
represented as a 4-axis multi-dimensional array.  Another way to look at it is,
that for any two vectors on a tangent space, it returns a linear transformation
that describes how the parallel transport deviates a vector.
Note: Since we have different tensor for each point, the curvature is a *local*
phenomenon.

The math for curvature is quite involved because we're not in flat space
anymore. This means we need to setup a lot of additional structures to deal
with the fact that we're moving tangent vectors with different tangent spaces.
Basically, I'm going to gloss over these details which are beyond the scope
of this post (and not to mention my current understanding) :p

Once we have this curvature tensor, we can come up with a measure of curvature
called the `sectional curvature <https://en.wikipedia.org/wiki/Sectional_curvature>`__
at a point :math:`P` denoted by :math:`K(u, v)`:

.. math::

   K(u, v) = \frac{\langle R(u, v)v, U\rangle }{\langle u, u\rangle \langle v, v\rangle  - \langle u, v\rangle ^2} \tag{1}

where :math:`u, v` are linearly independent vectors in the tangent space of
point :math:`P` and the angle brackets are the inner product.

|h3| Manifolds with Constant Sectional Curvature |h3e|

Riemannian manifolds with constant curvature at every point are special cases.
They come in three forms, constant:

* Constant Positive Curvature: Elliptic geometry
* Constant Zero Curvature: Euclidean geometry
* Constant Negative Curvature: Hyperbolic geometry

The first two we are more familiar with: The manifold model for Euclidean
geometry is just any Euclidean space.  The manifold model for elliptic geometry
is simply just a sphere.  The model for hyperbolic geometry is a bit more complicated
and we'll spend some more time with it in the next section.


.. admonition:: Euclidean and Non-Euclidean Geometries

  We saw above about how constant sectional curvature can be used to induce
  different types of geometries.  But what does this mean?  Let's dig in a bit.

  If we go way back to Euclid in his seminal work *Elements*, he gives `five
  postulates <https://en.wikipedia.org/wiki/Euclidean_geometry#Axioms>`__ (or axioms):

  1. A straight line segment can be drawn joining any two points.
  2. Any straight line segment can be extended indefinitely in a straight line.
  3. Given any straight line segment, a circle can be drawn having the segment as radius and one endpoint as center.
  4. All right angles are congruent.
  5. If two lines are drawn which intersect a third in such a way that the sum of the inner angles on one side is less than two right angles, then the two lines inevitably must intersect each other on that side if extended far enough. (aka the `parallel postulate <https://en.wikipedia.org/wiki/Parallel_postulate>`__ shown in Figure 5)

  .. figure:: /images/parallel_postulate.png
    :height: 200px
    :alt: Euclid's Parallel Postulate
    :align: center

    Figure 5: Euclid's Parallel Postulate: If two lines are placed over a third such that the sum of angles (:math:`\alpha+\beta`) are less than two right angles, then they must intersect at some point (source: Wikipedia).

  These postulates define Euclidean geometry which is a `axiomatic system
  <https://en.wikipedia.org/wiki/Axiomatic_system>`__.  An axiomatic system is
  one in which every theorem can be logically derived from it.  This is all
  jargon relating to logic. So the the standard geometry we learn in grade
  school with lines, cirles, angles, etc. is Euclidean geometry.

  However, this is not exactly the same thing as the analytical geometry 
  we study when we first learn about with Euclidean plane (i.e.  :math:`\mathbb{R}^2`).
  The Euclidean plane (and by extension Euclidean space) is a 
  `model <https://en.wikipedia.org/wiki/Axiomatic_system#Models>`__ of 
  Euclidean geometry:

      A model for an axiomatic system is a well-defined set, which assigns meaning for the undefined terms presented in the system, in a manner that is correct with the relations defined in the system.

  For example, Euclidean geometry defines a point but there is no concrete
  meaning to it.  In the Euclidean plane, we could define a point as a pair of
  real numbers :math:`(x,y)` with :math:`x,y \in \mathbb{R}`.  You could do the
  same with a line, circle and even an angle (in terms of the 
  `metric tensor <https://en.wikipedia.org/wiki/Metric_tensor#Length_and_angle>`__).
  Once we concretely define all these abstract things in Euclidean geometry,
  we have a **model** of Euclidean geometry.
  Non-Euclidean geometries can be defined by similar postulates.
  In particular, by modifying the parallel postulate, we can derive other types
  of geometries.  
  
  |h3| Elliptic Geometry |h3e|

  In Elliptic geometry, we change the parallel postulate to:

     Two lines perpendicular to a given line must intersect.

  where a visualization in shown in Figure 6.

  .. figure:: /images/elliptic_geometry.jpg
    :height: 300px
    :alt: Elliptic Geometry Postulate
    :align: center

    Figure 6: Two lines perpendicular to another line in Elliptic geometry, must intersect at some point (`source <http://www.math.cornell.edu/~mec/mircea.html>`__).

  A model of Elliptic geometry is manifold defined by the surface of a sphere
  (say with radius=1 and the appropriately induced metric tensor).  We can see that the
  Elliptic postulate holds, and it also yields different theorems than standard
  Euclidean geometry, such as the sum of angles in a triangle is greater than
  180 degrees.

  |h3| Hyperbolic Geometry |h3e|

  In Hyperbolic geometry, we change the parallel postulate to:

     For any given line R and point P not on R, in the plane containing both
     line R and point P there are at least two distinct lines through P that do
     not intersect R.

  where a visualization in shown in Figure 7.

  .. figure:: /images/hyperbolic_geometry.png
    :height: 200px
    :alt: Hyperbolic Geometry Postulate
    :align: center

    Figure 7: Lines :math:`x` and :math:`y` intersecting at :math:`P` never pass through line :math:`R`, although it is possible that they can asymptoptically approach it (`source <http://www.math.cornell.edu/~mec/mircea.html>`__).

  This figure is not a great visualization because, as we'll mention below, you
  can't really represent 2D hyperbolic geometry in 3D Euclidean space.  This makes
  it hard to visualize, and results in a more complex model than the other two
  geometries.  Further down, we'll describe a couple of models of hyperbolic
  geometry because there is no one "standard" model as we'll see below.


|h2| Hyperbolic Space |h2e|

`Hyperbolic space <https://en.wikipedia.org/wiki/Hyperbolic_space>`__
is a type of manifold with constant negative (sectional) curvature.
Formally:

  Hyperbolic n-space (usually denoted :math:`\bf H^n`), is a maximally symmetric,
  simply connected, n-dimensional Riemannian manifold with constant negative
  sectional curvature.

Hyperbolic space analogous to the n-dimensional sphere (which has constant
positive curvature).
This is very hard thing to visualize though because we're used to only 
imagining objects in Euclidean space -- not curved space.
One to think about it is that when embedded into Euclidean space, every point
is a `saddle point <https://en.wikipedia.org/wiki/Saddle_point>`__, still kind of
hard to imagine.
The real tough part is that even for the 2D hyperplane, we cannot embed it in
3D Euclidean space (`Hilbert's theorem <https://en.wikipedia.org/wiki/Hilbert%27s_theorem_(differential_geometry)>`__),
so something's got to give.

Fortunately, we can embed models of 2D hyperbolic geometry into a
pseudo-Euclidean space called Minkowski space.  Interestingly, Minkowski space
and curved space in general has applications in special relativity and general
relativity, respectively.  Let's see how this works.

|h3| Minkowski Space |h3e|

Before we get to the model of hyperbolic geometry, we have to set a few things up.
First, let's define a (generalized) `Minkowski space <https://en.wikipedia.org/wiki/Minkowski_space>`__
(a type of `pseudo-Riemannian manifold <https://en.wikipedia.org/wiki/Pseudo-Riemannian_manifold>`__):

    For :math:`n>=2`, a :math:`n`-dimensional Minkowski space is a real vector space
    of real dimension :math:`n` which there is a constant Minkowski metric of
    signature (n-1, 1) or (1, n-1).

Even though it's defined as a vector space, we can regard it as basically having
:math:`n` real dimensions (just like :math:`\mathbb{R}^n`) but with a special
type of metric tensor (generalization of dot product): the 
`Minkowski metric <https://en.wikipedia.org/wiki/Minkowski_space#Minkowski_metric>`__.
This is where it gets a bit "mind-bending".

The Minkowski metric is not too different from the standard Euclidean metric
tensor.  For two vectors :math:`{\bf u}=(u_1, \ldots, u_n)` and :math:`{\bf v} = (v, \ldots, v_n)`, 
we have:

.. math::

    g_E({\bf u, v}) &= u_1 v_1 + u_2 v_2 + \ldots + u_n v_n && \text{Euclidean Metric} \\
    g_M({\bf u, v}) &= \pm [u_1 v_1 - u_2 v_2 - \ldots - u_n v_n] && \text{Minkowski Metric} \\ \tag{2}
  
Notice that dimension 1 is treated differently [1]_ (alternatively dimension 2 to n
are different).  In `special relativity <https://en.wikipedia.org/wiki/Special_relativity>`__,
dimension 1 is considered the "time"-like dimension while the others are
"space"-like dimensions.  But I don't think this helps with the intuition all
that much.  For now, we just need to know that one of the dimensions is treated
differently, while the others are very similar to regular our Euclidean space.

|h3| Hyperboloid |h3e|

Before we get to the model, we need to cover the 
`hyperboloid <https://en.wikipedia.org/wiki/Hyperboloid>`__.
A hyperboloid is a
generalization of a hyperbola in two dimensions.  If you take a hyperbola and
spin it around its principal axes (or add certain types of affine
transformations), you get a hyperboloid.  There are a few other types of
hyperboloids but we'll only be talking about the two sheet version.  
Figure 8 shows an example of a two sheet hyperboloid.  

.. figure:: /images/hyperboloid.png
  :height: 300px
  :alt: Two Sheet Hyperboloid
  :align: center

  Figure 8: Example of two sheet hyperboloid (source: Wikipedia).


The two sheet hyperboloid (in three dimensions) has the following equation:

.. math::

    \frac{x^2}{a^2} + \frac{y^2}{b^2} - \frac{z^2}{c^2} = -1 \tag{3}

where with :math:`z > 0` defines the forward sheet.  You can see how
this can easily extend to multiple dimensions: you just add more positive
quadratic terms in front, with your "special" negative dimension (in this case
:math:`z`) in the back.

A common parametric representation with parameter :math:`t` of the two sheet
hyperboloid is:

.. math::

    x &= a\sinh t \cos\theta \\
    y &= b\sinh t \sin\theta \\
    z &= \pm c\cosh t \\
    \tag{4}

Notice that we're using the 
`hyperbolic functions <https://en.wikipedia.org/wiki/Hyperbolic_function>`__
here, which are very much related to hyperbolas (as expected).  Also notice
that we use :math:`\cos\theta` and :math:`\sin\theta` for :math:`x` and
:math:`y`, which reminds us of the parameterization of a circle.  Indeed, x and
y are the "normal" Euclidean dimensions and for a given constant :math:`z`,
they define a circle.


|h3| The Hyperboloid Model of Hyperbolic Geometry |h3e|

Finally we get to our first model of hyperbolic geometry!
Having introduced the two above concepts, our first model
known as the `hyperboloid model <https://en.wikipedia.org/wiki/Hyperboloid_model>`__
(aka Minkowski model or Lorentz model) is a model of n-dimensional
hyperbolic geometry in which points are represented on the forward sheet of a
two-sheeted hyperboloid of (n+1)-dimensional Minkowski space.
The points on this sheet are defined by:

.. math::

    x^2 + y^2 - z^2 = -1 \tag{5}

where :math:`z>0`.  Simple right?  Well in case it's not, let's try to add some
intuition.

So we know 2D Elliptic geometry (constant positive curvature) can be mapped to
a sphere in 3D Euclidean space.  So we can re-create any of our usual geometric
concepts and have them map to the sphere, for example, points lines, angles,
triangles etc.  In a similar way for hyperbolic geometry, we can have the same
concepts but mapped to the surface of the forward sheet of the hyperboloid,
*except* now we're in Minkowski space which makes your intuition about what
should happen all screwy.

For example, how does a straight line get defined? On a sphere
there are many paths you could take from point A to B, but the shortest path
"line" between two points is called a geodesic: the path you take between two
points when slicing through the center of the sphere  
(i.e. a `great circle <https://en.wikipedia.org/wiki/Great_circle>`__).
A `geodesic <https://en.wikipedia.org/wiki/Geodesic>`__ is the generalization
of a straight line to curved space, defined to be a curve where you can
parallel transport a tangent vector without deformation.

In our hyperboloid model, the geodesic (or our hyperbolic line) is defined to
be the curve created by intersecting two points and the origin with the
hyperboloid.  So you end up having to go "down" first then back up to reach a
point, never just directly towards it using the shortest path (on the surface)
in Euclidean space.  Figure 9 shows a visualization of this curve (ignore the
bottom circle for now, we'll come back to this later).

.. figure:: /images/hyperboloid_projection.png
  :height: 250px
  :alt: Two Sheet Hyperboloid
  :align: center

  Figure 9: Hyperboloid Stereoscopic Projection (source: Wikipedia).

Once we have the concept of a line, we can define it as the distance 
between two points.  This can be calculated using the 
`arc length <https://en.wikipedia.org/wiki/Arc_length>`__ of the tangent
vectors with the Minkowski metric.  This lends itself well to a closed form
for the hyperbolic distance between two points :math:`\bf u` and :math:`\bf v`:

.. math::

    d({\bf u, v}) = arcosh(g_M({\bf u, v})) \tag{6}

where we're using the inverse (or arc) hyperbolic cosine function and the
Minkowski metric defined above.


.. admonition:: Example: Calculating the Arc Length of a Geodesic In Hyperbolic Space

    To illustate a few of the above ideas and to gain some intuition, let's
    calculate the arc length of two points on the hyperbolic plane embedded in
    3D Minkowski space.  First, let's define a curve from A to B parameterized
    by :math:`t=[t_a, t_b]`, this is just a variation on Equation 4 where we set
    :math:`\theta=0`:

    .. math::

        x &= \sinh t  \\
        y &= 0  \\
        z &= \cosh t \\
        \tag{7}
   
    Computing the `arc length <https://en.wikipedia.org/wiki/Arc_length#Generalization_to_(pseudo-)Riemannian_manifolds>`__
    on a manifold, we have:

    .. math::

        d(A, B) &= \int_{t_a}^{t_b} \sqrt{
            g_M\big(  (\frac{dx(t)}{dt}, \frac{dy(t)}{dt}, \frac{dz(t)}{dt}),
                    (\frac{dx(t)}{dt}, \frac{dy(t)}{dt}, \frac{dz(t)}{dt})
            \big)} dt \\
        &= \int_{t_a}^{t_b} \sqrt{
               \big(\frac{dz(t)}{dt}\big)^2 - \big(\frac{dx(t)}{dt}\big)^2 - \big(\frac{dy(t)}{dt}\big)^2
            } dt \\
        &= \int_{t_a}^{t_b} \sqrt{ \big(\frac{d \cosh t}{dt}\big)^2 - \big(\frac{d \sinh t}{dt}\big)^2 } dt \\
        &= \int_{t_a}^{t_b} \sqrt{ \sinh^2 t -\cosh^2 t } dt \\
        &= \int_{t_a}^{t_b} \sqrt{ 1 } dt \\
        &= t_b - t_a \\
        \tag{8}

    This should match up to Equtaion 8.  Let's see what happens:

    .. math::

        d(A, B) &= arcosh\Big(g_M\big( 
            (\sinh t_a, 0, \cosh t_a),
            (\sinh t_b, 0, \cosh t_b),
        \big)  \Big) \\
        &= arcosh(\cosh t_a \cosh t_b - \sinh t_a \sinh t_b) \\
        &= arcosh(cosh(t_b - t_a)) && \text{hyperbolic identity} \\
        &= t_b - t_a \\
        \tag{9}

    As expected, the Equation 6 properly calculates the arc length with the
    Minkowski metric in this case.


* explain circles



|h3| Poincare Ball Model |h3e|

* Project hyperboloid using stereographic projection to unit circle
* Show metric, distance, etc.

* Include D3 visualization


|h2| Poincaré Embeddings for Hierarchical Representations |h2e|

* Explain algorithm (roughly)
* Explain *why* 

|h2| Poincaré Embeddings with NLP package (notebook?) |h2e|



|h2| Conclusion |h2e|



|h2| Further Reading |h2e|

* Previous posts: `Tensors, Tensors, Tensors <link://slug/tensors-tensors-tensors>`__, `Manifolds: A Gentle Introduction <link://slug/manifolds>`__
* Wikipedia: `Riemannian Manifold <https://en.wikipedia.org/wiki/Riemannian_manifold>`__, `Hyperbolic Space <https://en.wikipedia.org/wiki/Hyperbolic_space>`__, `Sectional Curvature <https://en.wikipedia.org/wiki/Sectional_curvature>`__, `Gaussian Curvature <https://en.wikipedia.org/wiki/Gaussian_curvature>`__, `parallel transport <https://en.wikipedia.org/wiki/Parallel_transport>`__
* [1] `Poincaré Embeddings for Learning Hierarchical Representations <https://arxiv.org/abs/1705.08039>`__, Maximilian Nickel, Douwe Kiela
* [2] `Implementing Poincaré Embeddings <https://rare-technologies.com/implementing-poincare-embeddings/>`__, Jayant Jain

.. [1] Both :math:`+` and :math:`-` make the math work out.  It's by convention you pick one or the other.  We'll be using a leading positive sign in this post.
