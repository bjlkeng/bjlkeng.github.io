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

|h3| Physical Vectors as Tensors |h3e|

We'll start with a concept we're all familiar with: vectors.
Now there are many different variants of vectors but we want to talk
about the physical vectors that has a magnitude and direction.  
In particular, we're *not* talking specifically about an ordered pair of
numbers (e.g. :math:`[1, 2]` in 2 dimensions).

Of course, we're all familiar with representing vectors as ordered pair
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

  Figure 1: The physical vector A (in red) is the same regardless of what basis
  you use (source: Wikipedia).

You can see in Figure 1 that we have a vector :math:`A` (in red) that can
be represented in two different basis: :math:`e^1, e^2` (blue) and :math:`e_1, e_2` (yellow) [1]_.  You can see it's the same old vector, it's just that the way we're describing it changed.  In the former case, we can describe it by :math:`[a_1, a_2]`,
while in the latter by :math:`[a^1, a^2]` (note: the super/subscripts represent
different values, which we'll get to, and you can ignore all the other stuff in
the diagram).

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
        = \begin{bmatrix} cos(\frac{pi}{2}) & -sin(\frac{pi}{2}) \\ 
                        sin(\frac{pi}{2}) & cos(\frac{pi}{2}) \end{bmatrix}
        = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                          \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
        \tag{5}

    The `inverse <https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_2_%C3%97_2_matrices>`__ 
    of our rotation matrix is:

    .. math::

        \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix}^{-1}
        = \frac{1}{(\frac{1}{\sqrt{2}})(\frac{1}{\sqrt{2}}) - (-\frac{1}{\sqrt{2}})(\frac{1}{\sqrt{2}})}
           \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} 
        = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} 
        \tag{6}
        
    Therefore our vector :math:`v` can be represented in this basis 
    (:math:`[\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}], [-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}]`)
    as:

    .. math::

        = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                          \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} 
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
it is said to be a *rank=1* or *order=1* tensor.

In physics and other domains, one may want to work in a non-Euclidean basis
because it's more convenient, but the objects


- can be represented by a 1-d array but it's *not* the 1-d array
- (height, width, length) as an example of non-tensor
- different coordinate system, yields different 
- examples, pictures
- transformation independent, transform *against* basis transform

.. admonition:: Example 2: Non-Tensors

    Using our definition of a vector with magnitude length, we get
    the idea that that tensors represent some kind of object that is invariant
    to a basis change.  Every tensor can be represented as a 
    `coordinate vector <https://en.wikipedia.org/wiki/Coordinate_vector>`__
    in particular basis but every tuple of 3 numbers is *not* a tensor.

    For example, we can represent the height, width and length of a box as an
    ordered list of numbers: :math:`[10, 20, 15]`.  However, this is not a
    tensor because if we change our coordinate system, the height, width and length
    of the box don't change, they stay the same.  Therefore, this tuple
    is not a tensor.



|h3| Covariant vs. Contravariant Tensors |h3e|

- some vectors transform *with* and some against
- gradient (velocity) vs. physical vectors

|h3| Einstein Notation for Tensors |h3e|

- Just some convenience notation
- Explain "up" and "down"
- Explain "missing indices" etc.

|h3| Examples of Common Tensors |h3e|

- Vector, 
- Covector, linear functional
- Dot Product
- Linear transformation


|h3| Summary: A Tensor is a Tensor |h3e|

- Summarize high-level point of tensors


|h2| Manifolds |h2e|

|h2| Metric Spaces |h2e|

|h2| Conclusion |h2e|

|h2| Further Reading |h2e|


* Wikipedia: `Tensors <https://en.wikipedia.org/wiki/Tensor_(disambiguation)>`__,
  `Manifold <https://en.wikipedia.org/wiki/Manifold>`__,
  `Metric Space <https://en.wikipedia.org/wiki/Metric_space>`__,
  `Covariance and contravariance of vectors <https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors>`__,
  `Vector <https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics)>`__

.. [1] This is not exactly the best example because it's showing a vector in both contravariant and tangent covector space, which is not exactly the point I'm trying to make here.  But the idea is basically the same: the vector is the same object regardless of what basis you use.
