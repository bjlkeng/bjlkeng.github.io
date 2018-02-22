.. title: Tensors, Manifolds and Metric Spaces
.. slug: tensors-manifolds-and-metric-spaces
.. date: 2018-02-15 07:24:57 UTC-05:00
.. tags: tensors, manifolds, metric spaces, metrics, mathjax
.. category: 
.. link: 
.. description: A quick introduction to tensors, manifolds and metrics spaces.
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

This post is going to take a step back from some of the machine learning
topics that I've been writing about recently and go back to some basics: math!
In particular, tensors, manifolds and metric spaces.  These are all topics that
are casually mentioned in machine learning papers but for those of us who never
took advanced geometry or topology courses (*cough* computer engineers), it's a
bit murky trying to understand what's going on.  
So on my most recent vacation, I started reading a variety of sources on the
interweb trying to piece together a picture of what these topics were all
about.  As usual, I'll skip the formalities (partly because I probably couldn't
do them justice) and try to explain the intuition instead.  I'll sprinkle in
a bunch of examples and also try to relate it back to ML where possible.
Hope you like it!

.. TEASER_END

|h2| Tensors |h2e|

|h3| A Tensor by Any Other Name |h3e|

For newcomers to ML, tensor have to be one of the top ten confusing things
to come across.  Not only because the term is new, but also because it's used
ambiguously with other branches of mathematics and physics!  In ML, it's
often colloquially as a multidimensional array.  That's what people usually mean
when they talk about "tensors" in the context of things like TensorFlow.
However, tensors as multidimensional arrays is just one very narrow "view" of a
tensor, tensors (mathematically speaking) are much more than that!
Let's start at the beginning.

|h3| Geometric Vectors as Tensors |h3e|

We'll start with a concept we're all familiar with: `geometric vectors <https://en.wikipedia.org/wiki/Euclidean_vector>`__ (also called Euclidean vectors).
Now there are many different variants of vectors but we want to talk specifically
about the geometric vectors that have a magnitude and direction.  
In particular, we're *not* talking about just an ordered pair of numbers (e.g.
:math:`[1, 2]` in 2 dimensions).

Of course, we're all familiar with representing vectors as ordered pairs
but recall that's probably because we're just *assuming* that we're working in
Euclidean space where each of the indices represent the component of the basis
vector (e.g. :math:`[1, 0]` and :math:`[0, 1]` in 2 dimensions).
If we change basis to some other 
`linearly independent <https://en.wikipedia.org/wiki/Linear_independence>`__
basis, the components will definitely change, but will the magnitude and
direction change?  No!  It's still the same old vector with a magnitude and
direction. 
When changing basis, we're just describing or "viewing" the vector in a different
way but fundamentally it's still the same old vector.
Figure 1 shows a visualization.
 

.. figure:: /images/vector_tensor.png
  :height: 250px
  :alt: A Physical Vector
  :align: center

  Figure 1: The geometric vector A (in red) is the same regardless of what basis
  you use (source: Wikipedia).

You can see in Figure 1 that we have a vector :math:`A` (in red) that can
be represented in two different basis: :math:`e^1, e^2` (blue) and :math:`e_1, e_2` (yellow) [1]_.  You can see it's the same old vector, it's just that the way we're describing it changed.  In the former case, we can describe it 
as a `coordinate vector <https://en.wikipedia.org/wiki/Coordinate_vector>`__
by :math:`[a_1, a_2]`,
while in the latter by the coordinate vector :math:`[a^1, a^2]` (note: the
super/subscripts represent different values, which we'll get to, and you can
ignore all the other stuff in the diagram).

So then a vector is the geometric object, *not* specifically its representation
in a particular basis.

.. admonition:: Example 1: A vector in different basis.

    Let's take the vector :math:`v` as :math:`[a, b]` in the standard Euclidean
    basis: :math:`[1, 0]` and :math:`[0, 1]`.  Another way to write this is as:

    .. math::

        v = a \begin{bmatrix} 1 \\ 0 \end{bmatrix}
          + b \begin{bmatrix} 0 \\ 1 \end{bmatrix}
        \tag{1}

    Now what happens if we `scale <https://en.wikipedia.org/wiki/Scaling_(geometry)>`__
    our basis by :math:`2`?  This can be represented by multiplying our basis matrix
    (where each column is a one of our basis vectors) by a transformation matrix:

    .. math::

        \text{original basis} * \text{scaling matrix}
        =
        \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} 
        \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} 
        = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} 
        \tag{2}
        
    So our new basis is :math:`[2, 0]` and :math:`[0, 2]`.  But how does our
    original vector :math:`v` get transformed?  We actually have to multiply
    by the inverse scaling matrix:

    .. math::

        v = 
        \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}^{-1}
        \begin{bmatrix} a \\ b \end{bmatrix} 
        =
        \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & \frac{1}{2} \end{bmatrix} 
        \begin{bmatrix} a \\ b \end{bmatrix} 
        = \begin{bmatrix} \frac{a}{2} \\ \frac{b}{2} \end{bmatrix} 
        \tag{3}

    So our vector is represented as :math:`[\frac{a}{2}, \frac{b}{2}]` in our
    new basis. We can see results in the exact same vector regardless of
    what basis we're talking about:

    .. math::

        v = a \begin{bmatrix} 1 \\ 0 \end{bmatrix}
          + b \begin{bmatrix} 0 \\ 1 \end{bmatrix}
          = \frac{a}{2} \begin{bmatrix} 2 \\ 0 \end{bmatrix}
          + \frac{b}{2} \begin{bmatrix} 0 \\ 2 \end{bmatrix}
        \tag{4}

    |hr|

    Now let's do a more complicated transform on our Euclidean basis.  Let's
    `rotate <https://en.wikipedia.org/wiki/Rotation_matrix>`__ 
    the axis by 45 degrees, the transformation matrix is this:

    .. math::

        \text{rotation matrix} 
        = \begin{bmatrix} cos(\frac{\pi}{2}) & -sin(\frac{\pi}{2}) \\ 
                        sin(\frac{\pi}{2}) & cos(\frac{\pi}{2}) \end{bmatrix}
        = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                          \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
        \tag{5}

    The `inverse <https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_2_%C3%97_2_matrices>`__ 
    of our rotation matrix is:

    .. math::

        \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix}^{-1}
        = \frac{1}{(\frac{1}{\sqrt{2}})(\frac{1}{\sqrt{2}}) - (-\frac{1}{\sqrt{2}})(\frac{1}{\sqrt{2}})}
           \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} 
        = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} 
        \tag{6}
        
    Therefore our vector :math:`v` can be represented in this basis 
    (:math:`[\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}], [-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}]`)
    as:

    .. math::

        \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                          -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} 
        \begin{bmatrix} a \\ b  \end{bmatrix} 
        = \begin{bmatrix} \frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}} \\ 
                          \frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}  \end{bmatrix} \\
        \tag{7}

    Which we can see is exactly the same vector as before:

    .. math::
        
        v = a \begin{bmatrix} 1 \\ 0 \end{bmatrix}
          + b \begin{bmatrix} 0 \\ 1 \end{bmatrix}
          = (\frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}}) 
            \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}
          + (\frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}) 
            \begin{bmatrix} \frac{-1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}
        \tag{8}   


So Example 1 shows us how a vector represents the same thing regardless of 
what basis you happen to be working in.  As you might have guessed,
these physical vectors are tensors!  Since it has one physical axis,
it is said to be a *rank=1* tensor.  A scalar is said to be a *rank=0* tensor,
which is just pretty much just a degenerate case.


In physics and other domains, you may want to work in a non-Euclidean basis
because it's more convenient, but still want to talk about the same objects
regardless if we're in Euclidean basis or not.

So geometric vectors are our first step in understanding tensors.  To summarize
some of the main points:

* Tensors can be viewed as an ordered list of numbers with respect to a basis
  but that isn't the tensor itself.
* They are independent of a change in basis (i.e. their representation changes
  but what they represent does not)
* The *rank* (or *degree* or *order*) of a tensor specifies how many axis you
  need to specify it (careful this is different than the dimensional space
  we're working in)

Just to drive the first point home, Example 2 shows an example of a tuple
that might look like it represents a tensor but is not.

.. admonition:: Example 2: Non-Tensors

    We can represent the height, width and length of a box as an ordered list
    of numbers: :math:`[10, 20, 15]`.  However, this is not a tensor because if
    we change our coordinate system, the height, width and length of the box
    don't change, they stay the same.  Tensors, however, have specific rules
    of how to change their representation when the basis changes.
    Therefore, this tuple is not a tensor.


|h3| Covariant vs. Contravariant Tensors |h3e|

In the last section, we saw how geometric vectors as tensors are invariant to
basis transformations and how you have to multiply the inverse of the basis
transformation matrix with the coordinates in order to maintain that invariance
(Example 1).  Well it turns out depending on the type of tensor, how you
"maintain" the invariance can mean different things.

A geometric vector is an example a **contravariant** vector because when
changing basis, the components of the vector transform with the inverse of the
basis transformation matrix (Example 1).  It's easy to remember it as
"contrary" to the basis matrix.  As convention, we will usually label
contravariant vectors with a superscript and write them as column vectors:

.. math::

    
    v^\alpha = \begin{bmatrix} v^0 \\ v_1 \\ v_2 \end{bmatrix}  \tag{9}

In Equation 9, :math:`\alpha` is *not* an exponent, instead we should think
of it as a "loop counter", e.g. :math:`\text{for } \alpha \text{ in } 0 .. 2`.
Similarly, the superscripts inside the vector correspond to each of the
components in a particular basis, indexing the particular component.
We'll see a bit later why this notation is convenient.

As you might have guessed, the other type of vector is a **covariant** vector
(or **covector** for short) because when changing basis, the components of the
vector transform with the *same* basis transformation matrix.  
You can remember this one because it "co-varies" with the basis transformation.
As with contravariant vectors, a covector is a tensor of rank 1.
As convention, we will usually label covectors with a subscript and write them
as a row vectors:

.. math::

    u_\alpha = [ v_0, v_1, v_2 ]   \tag{10}

Now covectors are a little bit harder to explain than contravariant vectors
because the examples of them are more abstract than geometric vectors [2]_.
Firstly, they do *not* represent geometric vectors (or else they'd be
contravariant).  Instead, we should think of them as a linear function that
takes a vector as input (in a particular basis) and maps it to a scalar, i.e.:

.. math::

    f({\bf x}) = v_0 x_0 + v_1 x_1 + v_2 x_2 \tag{11}

This is an important idea: a contravariant vector is an object that has an
input (vector) and produces an output (scalar), independent of the basis you
are in.  This is a common theme we'll see in tensors: input, output, and
independent of basis.  Let's take a look an example of how they arise.


.. admonition:: Example 3: A differential as a Covariant Vector

    Let's define a function and its differential in :math:`\mathbb{R}^2` in the
    standard Euclidean basis:
    
    .. math::

        f(x,y) &= x^2 + y^2 \\
        df &= 2x dx + 2y dy \tag{12}

    If we are given a fixed point :math:`(x_0,y_0) = (1,2)`, then the differential
    evaluated at this point is:

    .. math::

        df_{(x_0,y_0)} &= 2(1) dx + 2(2) dy \\
                    &= 2dx + 4dy  \\
        g(x, y) &:= 2x + 4y  \\ \tag{13}

    where in the last equation, I just relabelled things in terms of :math:`g,
    x, \text{ and } y`, which looks exactly like a linear functional!

    As we would expect with a tensor, the "behavior" of this covector shouldn't
    really change even if we change basis.  If we evaluate this functional
    at a geometric vector :math:`v=(a, b)`, then of course we get
    :math:`g(a,b)=2a + 4b`, a scalar.  If this truly is a tensor, this scalar
    should not change even if we change our basis.

    Let's rotate the axis 45 degrees.  From example 1, we know the rotation matrix
    and the inverse of it:

    .. math::

        R := \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                          \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix},
        \text{ }
        R^{-1} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
        \tag{14}

    To rotate our original point :math:`(a,b)`, we multiply the inverse matrix
    by the column vector as in Equation 7 to get :math:`v` in our new basis,
    which we'll denote by :math:`v_{R}`:

    .. math::

        v_{R} = \begin{bmatrix} \frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}} \\ 
                        \frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}  \end{bmatrix} \\
        \tag{15}

    If you believe what I said before about covectors varying with the basis
    change, then we should just need to multiple our covector, call it
    :math:`u` (as a column vector in the standard Euclidean basis) by our
    transformation matrix:

    .. math::

        u_{R} = u * R &= [2, 4] \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
            &= [3\sqrt{2}, \sqrt{2}] \\
            \tag{16}

    Evaluating v_R at u_r:

    .. math::

        u_R (v_R) &= 3\sqrt{2} (\frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}})
                   + \sqrt{2} (\frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}) \\
                  &= 3a + 3b - a + b \\
                  &= 2a + 4b

    which is precisely the scalar that we got in the Euclidean basis. 

Before we move on, I want to introduce some more notation to simply our lives.
From Equation 11, using some new notation, we can re-write it covector
:math:`u_\alpha` with input geometric vector :math:`v^\alpha` (specified by
their coordinates in the same basis) as:

.. math::

    <u_\alpha, v^\alpha> = \sum_{\alpha=0}^2 u_\alpha v^\alpha
    = u_0 v^0 + u_1 v^1 + u_2 v^2 = u_\alpha v^\alpha \tag{12}

Note as before the superscripts are *not* exponentials but rather denote
an index.
The last expression uses the **Einstein summation convention**: if the
same "loop variable" appear once in both a lower and upper index, it means to
implicitly sum over that variable.  This is standard notation in physics
textbooks and makes the tedious step of writing out summations much easier.
Also note that covectors have a subscript and contravariant vectors have a
superscript.  This becomes more important as we deal with higher order tensors.

One last notational point is that we now know of two types of rank 1 tensors:
contravariant vectors (e.g. geometric vector) and covectors (or linear
functional).  Since they're both rank 1, we need to be a bit more precise.
We'll usually write of a :math:`(n, m)`-tensor where :math:`n` is the 
contravariant component and :math:`m` is the covariant component.  The rank
is the sum of :math:`m+n`.  Therefore a contravariant vector is a
:math:`(1, 0)`-tensor and a covector is a :math:`(0, 1)`-tensor.


|h3| Linear Transformations as Tensors |h3e|

- Vector, 
- Covector, linear functional
- Dot Product
- Linear transformation

|h3| Bilinear Forms and the Metric Tensor |h3e|

- Vector, 
- Covector, linear functional
- Dot Product
- Linear transformation
- Metric tensor: defines length and angle independent of basis


.. admonition:: Covector Basis
    
    When first learning linear algebra, there really didn't seem to be a
    difference between row vectors and columns vectors.  There basically
    looked like the same thing just flipped around.

    Well it turns out with the metric tensor, they basically have this
    "flipped" around relationship.

    Any geometric vector can be represented in its regular basis, but also in a
    covector basis.  :math:`\alpha(w) = g(v, w)`
    `see here <https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors#Covariant_transformation>`__

|h3| Summary: A Tensor is a Tensor |h3e|

- Summarize high-level point of tensors
- Table of all the tensors we've looke dat

|h2| Metric Tensors |h2e|

- definition of metric as distance

|h2| Manifolds |h2e|

- R^2 as manifold
- lines and circles
- Riemannian manifold: tangent space at each point with the inner product
- Defines arc length, thus distance


|h2| Conclusion |h2e|

|h2| Further Reading |h2e|


* Wikipedia: `Tensors <https://en.wikipedia.org/wiki/Tensor_(disambiguation)>`__,
  `Manifold <https://en.wikipedia.org/wiki/Manifold>`__,
  `Metric Space <https://en.wikipedia.org/wiki/Metric_space>`__,
  `Covariance and contravariance of vectors <https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors>`__,
  `Vector <https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics)>`__
* `Tensors for Laypeople <http://www.markushanke.net/tensors-for-laypeople/>`__, Markus Hanke
* `Tensors for Beginners (YouTube playlist) <https://www.youtube.com/playlist?list=PLJHszsWbB6hrkmmq57lX8BV-o-YIOFsiG>`__, eigenchris


.. [1] This is not exactly the best example because it's showing a vector in both contravariant and tangent covector space, which is not exactly the point I'm trying to make here.  But the idea is basically the same: the vector is the same object regardless of what basis you use.

.. [2] There is a `geometric interpretation of covectors <https://en.wikipedia.org/wiki/Linear_form#Visualizing_linear_functionals>`__ are parallel surfaces and the contravariant vectors "piercing" these surfaces.  I don't really like this interpretation because it's kind of artificial and doesn't have any physical analogue that I can think of.
