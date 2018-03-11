.. title: Tensors, Manifolds and Metric Spaces
.. slug: tensors-manifolds-and-metric-spaces
.. date: 2018-02-24 07:24:57 UTC-05:00
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

|h2| 1. Tensors |h2e|

|h3| 1.1 A Tensor by Any Other Name |h3e|

For newcomers to ML, tensor have to be one of the top ten confusing things
to come across.  Not only because the term is new, but also because it's used
ambiguously with other branches of mathematics and physics!  In ML, it's
often colloquially as a multidimensional array.  That's what people usually mean
when they talk about "tensors" in the context of things like TensorFlow.
However, tensors as multidimensional arrays is just one very narrow "view" of a
tensor, tensors (mathematically speaking) are much more than that!
Let's start at the beginning.

(By the way, you should checkout [2], which is a great series of videos
explaining tensors from the beginning.  It definitely helped clarify a lot of
ideas for me and a lot of this section is based on his presentation.)

|h3| 1.2 Geometric Vectors as Tensors |h3e|

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
which is just pretty much just a degenerate case.  Note: rank is different
than dimension.

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


|h3| 1.3 Covariant vs. Contravariant Tensors |h3e|

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
From Equation 11, using some new notation, we can re-write covector
:math:`u_\alpha` with input geometric vector :math:`v^\alpha` (specified by
their coordinates in the same basis) as:

.. math::

    <u_\alpha, v^\alpha> = \sum_{\alpha=0}^2 u_\alpha v^\alpha
    = u_0 v^0 + u_1 v^1 + u_2 v^2 = u_\alpha v^\alpha \tag{17}

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


|h3| 1.4 Linear Transformations as Tensors |h3e|

Another familiar transformation that we see is a 
`linear transformation <https://en.wikipedia.org/wiki/Linear_map>`__
(also called a linear map).  Linear transformations are just
like we remember from linear algebra, we can view them as matrices.
*But* a linear transformation is still the same linear transformation
when we change basis so it is also a tensor (with a matrix view being one view
of it)!

Let's review a linear transformation:

    A function :math:`L:{\bf u} \rightarrow {\bf v}` is a linear map if for any two vectors
    :math:`\bf u, v` and any scalar `c`, the following two conditions are
    satisfied (linearity):

    .. math::
        L({\bf u} + {\bf v}) &= L({\bf u}) + L({\bf v}) \\
        L(c{\bf u}) &= cL({\bf u})
        \tag{18}

One key idea here is that a linear transformation takes a vector :math:`\bf v`
to another vector :math:`L(\bf v)` *in the same basis*.  The linear transformation
itself has nothing to do with the basis (we of course can apply it to a basis).
Even though the "output" is a vector, it's analogous to the covectors (or linear functionals)
we saw above: an object that acts on a vector and returns something, independent of basis.

Okay, so what kind of tensor is this?  Let's try to derive it!
Let's suppose we have a geometric vector :math:`\bf v` and its transformed
output :math:`w = L{\bf v}` in an original basis, where :math:`L` is our linear
transformation (we'll use matrix notation here).
After some change in basis via a transform :math:`T`,
we'll end up with the same vector in the new basis :math:`\bf \tilde{v}` 
and the corresponding transformed version :math:`\tilde{w} = \tilde{L}{\bf \tilde{v}}`.
Note that since we're in a new basis, we have to use a new view of :math:`L`,
which we label as :math:`\tilde{L}`.

.. math::

    \tilde{L}{\bf \tilde{v}} &= \tilde{w} \\
    &= T^{-1}w  && \text{w is contravariant} \\ 
    &= T^{-1}L{\bf v}  && \text{definition of }w \\ 
    &= T^{-1}LT\tilde{\bf v}  && \text{since } {\bf v} = T\tilde{\bf v} \\ 
    \therefore \tilde{L}& = T^{-1}LT \\
    \tag{19}

The second last line comes from the fact that we're going from the new basis to the old
basis so we use the inverse of the inverse -- the original basis transform.

Equation 19 tells us something interesting, we're not just multiplying by the 
inverse transform (contravariant), nor just the forward transform (covariant),
we're doing both, which hints that this is a (1,1)-tensor!  Indeed, this is
our first example of a rank 2 tensor, which usually is represented as a matrix
(e.g. 2 axis).


.. admonition:: Example 4: A Linear Transformation as a (1,1)-Tensor

    Let's start with a simple linear transformation in our standard
    Euclidean basis:

    .. math::

        L = \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & 2 \end{bmatrix}
        \tag{20}

    Next, let's use the same 45 degree rotation for our basis as Example 1 and 2
    (which also happens to be a linear transformation):

    .. math::

        R := \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                          \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix},
        \text{ }
        R^{-1} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
        \tag{21}

    Suppose we're applying :math:`L` to a vector :math:`{\bf v}=(a, b)`, and then changing
    it into our new basis.  Recall, we would first apply :math:`L`, then apply
    a contravariant (inverse matrix) transform to get to our new basis:

    .. math::

        R^{-1}(L{\bf v}) &= \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix}
        \Big(\begin{bmatrix} \frac{1}{2} & 0 \\ 0 & 2 \end{bmatrix}
         \begin{bmatrix} a \\ b \end{bmatrix}\Big) \\
        &=\begin{bmatrix} \frac{a}{2\sqrt{2}} + \sqrt{2}b \\ -\frac{a}{2\sqrt{2}} + \sqrt{2}b \end{bmatrix}
        \tag{22}
        
    Equation 7 tells use what :math:`\tilde{\bf v} = R^{-1}{\bf v}` is in our new basis:

    .. math:: 
        \tilde{\bf v} = \begin{bmatrix} \frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}} \\ 
                        \frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}  \end{bmatrix}  \tag{23}

    Applying Equation 19 to :math:`L` gives us:

    .. math::

        \tilde{L} &= R^{-1}LR \\ 
        &= 
        \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} 
        \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & 2 \end{bmatrix}
        \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
        &= \begin{bmatrix} \frac{5}{4} & \frac{3}{4} \\ 
                        \frac{3}{4} & \frac{5}{5}  \end{bmatrix}\\
        \tag{24}

    Applying :math:`\tilde{L}` to  :math:`\tilde{\bf v}`:

    .. math::
        \tilde{L}\tilde{\bf v} &=
        \begin{bmatrix} \frac{5}{4} & \frac{3}{4} \\ 
                        \frac{3}{4} & \frac{5}{5}  \end{bmatrix}
        \begin{bmatrix} \frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}} \\ 
                        \frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}  \end{bmatrix} \\
        &= \begin{bmatrix} \frac{5a}{4\sqrt{2}} + \frac{5b}{4\sqrt{2}} 
                          - \frac{3a}{4\sqrt{2}} + \frac{3b}{4\sqrt{2}} \\
                            \frac{3a}{4\sqrt{2}} + \frac{3b}{4\sqrt{2}} 
                          - \frac{5a}{4\sqrt{2}} + \frac{5b}{4\sqrt{2}} 
         \end{bmatrix} \\
        &=\begin{bmatrix} \frac{a}{2\sqrt{2}} + \sqrt{2}b \\ -\frac{a}{2\sqrt{2}} + \sqrt{2}b \end{bmatrix} \\
        \tag{25}

    which we can see is the same as Equation 22.


|h3| 1.5 Bilinear Forms |h3e|

We'll start off by introducing a non-so-familiar idea (at least by name)
called the *bilinear form*.  Let's take a look at the definition with
respect to vector spaces:

    A function :math:`B:{\bf u, v} \rightarrow \mathbb{R}` is a bilinear form for two
    input vectors :math:`\bf u, v`, if for any other vector :math:`\bf w` and
    scalar `\lambda`, the following conditions are satisfied (linearity):

    .. math::
        B({\bf u} + {\bf w}, {\bf v}) &= B({\bf u}, {\bf v}) + B({\bf w}, {\bf v}) \\
        B(\lambda{\bf u}, {\bf v}) &= \lambda B({\bf u}, {\bf v})\\
        B({\bf u}, {\bf v} + {\bf w}) &= B({\bf u}, {\bf v}) + B({\bf u}, {\bf w}) \\
        B({\bf u}, \lambda{\bf v}) &= \lambda B({\bf u}, {\bf v})\\
        \tag{26}

All this is really saying is that we have a function that maps two geometric
vectors to the real numbers, and that it's "linear" in both its
inputs (separately, not at the same time) , hence the name "bilinear".  So
again, we see this pattern: a tensor takes some input and maps it to output
that in independent of a change in basis.

Similar to linear transformations, we can represent bilinear forms as a matrix
:math:`A`:

.. math::

    B({\bf u}, {\bf v}) = {\bf u^T}A{\bf v} = \sum_{i,j=1}^n a_{i,j}u_i v_j = A_{i,j}u^iv^j \tag{27}

where in the last expression I'm using Einstein notation to indicate that :math:`A`
is a rank (0, 2)-tensor, :math:`{\bf u, v}` are both (1, 0)-tensors (contravariant).

So let's see how we can show that this is actually a (0, 2)-tensor (two
covector components).  We should expect that when changing basis we'll need to
multiply by the basis transform twice ("with the basis"), along the same lines
as the linear transformation in the previous section, except with two covector
components now.
We'll use Einstein notation here, but you can check out Appendix A for the equivalent
matrix multiplication operation.

Let :math:`B` be our bilinear, :math:`u, v` geometric vectors, :math:`T` our basis
transform, and :math:`\tilde{B}, \tilde{u}, \tilde{v}` our post-transformed
bilinear and vectors, respectively.  Here's how we can show that the bilinear
transforms like a (0,2)-tensor:

.. math::

    \tilde{B}_{ij}\tilde{u}^i\tilde{v}^j &= B_{ij}u^iv^j && \text{output scalar same in any basis} \\
    &= B_{ij}T_k^i \tilde{u}^i T^j_l \tilde{v}^j && u^i=T^i_k \tilde{u}^k \\
    &= B_{ij}T_k^i T^j_l \tilde{u}^i \tilde{v}^j && \text{re-arrange summations}\\
    \therefore \tilde{B}_{ij}& = B_{ij}T_k^i T^j_l \\
   \tag{28} 

As you can see we transform "with" the change in basis, so we get a (0, 2)-tensor.
Einstein notation is also quite convenient (once you get used to it)!

|h3| 1.6 The Metric Tensor |h3e|

Before we end off on the tensor section, I want to introduce you to one of the
most important tensors around: the `Metric Tensor <https://en.wikipedia.org/wiki/Metric_tensor>`__.
In fact, it's probably one of the top reasons people start to learn about tensors.

The definition is a lot simpler because it's just a special kind of bilinear:

    A metric tensor at a point :math:`p` is a function :math:`g_p({\bf x}_p, {\bf y}_p)`
    which takes a pair of (tangent) vectors :math:`{\bf x}_p, {\bf y}_p` at :math:`p`
    and produces a real number such that:

    * :math:`g_p` is bilinear (see previous definition)
    * :math:`g_p` is symmetric: :math:`g_p({\bf x}_p, {\bf y}_p) = g_p({\bf y}_p, {\bf x}_p)`
    * :math:`g_p` is nondegenerate.  For every :math:`{\bf x_p} \neq 0` there exists
      :math:`{\bf y_p}` such that :math:`g_p({\bf x_p}, {\bf y_p}) \neq 0`

Don't worry so much about the "tangent" part, we'll get to that when we discuss manifolds below.

The metric tensor is important because it helps us (among other things) define
distance and angle between two vectors in a basis independent manner.  In the
simplest case, it's exactly our good old dot product operation for Euclidean
space.  But of course, we want to generalize this concept a little bit so we still
have the same "operation" under a change of basis -- that the resultant scalar
we produce should be the same.  Let's take a look.

In Euclidean space, the dot product (whose generalization is called the 
`inner product <https://en.wikipedia.org/wiki/Inner_product_space>`__) 
for two vectors :math:`{\bf u}, {\bf v}` is defined as:

.. math::

    {\bf u}\cdot{\bf v} = \sum_{i=1}^n u_i v_i \tag{29}

However, this can we re-written as for metric tensor :math:`g`:

.. math::

    {\bf u}\cdot{\bf v} = g({\bf u},{\bf v}) = g_{ij}u^iv^j 
        = [u^1 u^2] 
        \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} 
        \begin{bmatrix} v^1 v^0 \end{bmatrix} \tag{30}

where in the last expression I substituted the metric tensor in standard Euclidean space.
That is, the metric tensor in the standard Euclidean basis is just the identity
matrix:

.. math::

    g_{ij} = I_n \tag{31}

So now that we have a dot-product-like operation, we can define our
basis-independent definition of length of a vector, distance between two
vectors and angle between two vectors:

.. math::

    ||u|| = \sqrt{g_{ij} u^i u^j} \\
    d(u, v) = \sqrt{g_{ij} u^i v^j} \\
    cos(\theta) = \frac{g_{ij} u^i v^j}{||{\bf u}|| ||{\bf v}||} \\
    \tag{32}

The next example shows that the distance and angle are truly invariant between
a change in basis if we use our new metric tensor definition.


.. admonition:: Example 5: Computing Distance and Angle with the Metric Tensor

    Let's begin by defining two vectors in our standard Euclidean basis:

    .. math::

        {\bf u} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, 
        {\bf v} = \begin{bmatrix} 2 \\ 0 \end{bmatrix} \tag{33}

    Using our standard method for computing distance and angle:

    .. math::

        d({\bf u}, {\bf v}) &= \sqrt{({\bf u - v})({\bf u - v})} = \sqrt{(2 - 1)^2 + (0 - 1)^2}  = \sqrt{2} \\
        cos(\theta) &= \frac{{\bf u}\cdot {\bf v}}{||{\bf u}|| ||{\bf v}||} = \frac{2(1) + 1(0)}{(\sqrt{1^2 + 1^2})(\sqrt{2^2 + 0^2})} = \frac{1}{\sqrt{2}}  \\
        \theta &= 45^{\circ}
        \tag{34}

    Now, let's try to change our basis.  To show something a bit more
    interesting than rotating the axis, let's try to change to a basis
    of :math:`(2, 1)` and :math:`(-\frac{1}{2}, \frac{1}{4})`.  To  
    change basis (from a standard Euclidean basis), the transform we need to
    apply is:

    .. math::

        T = \begin{bmatrix} 2 & -\frac{1}{2} \\ 1 & \frac{1}{4} \end{bmatrix}, 
        T^{-1} = \begin{bmatrix} \frac{1}{4} & \frac{1}{2} \\ -1 & 2 \end{bmatrix}
        \tag{35}

    As you can see, it's just stacking the column vectors of our new basis
    side-by-side in this case (when transforming from a Euclidean space).  With
    these vectors, we can transform our :math:`{\bf u}, {\bf v}` to the new
    basis vectors :math:`\tilde{\bf u}, \tilde{\bf v}` as shown:

    .. math::

        \tilde{\bf u} &= T^{-1} {\bf u} = 
                \begin{bmatrix} \frac{1}{4} & \frac{1}{2} \\ -1 & 2 \end{bmatrix}
                \begin{bmatrix} 1 \\ 1 \end{bmatrix}
            = \begin{bmatrix} \frac{3}{4} \\ 1 \end{bmatrix} \\
        \tilde{\bf v} &= T^{-1} {\bf v} = 
                \begin{bmatrix} \frac{1}{4} & \frac{1}{2} \\ -1 & 2 \end{bmatrix}
                \begin{bmatrix} 2 \\ 0 \end{bmatrix}
            = \begin{bmatrix} \frac{1}{2} \\ -2 \end{bmatrix}
        \tag{36}

    Before we move on, let's see if using our standard Euclidean distance function 
    will work in this new basis:

    .. math::

        \sqrt{({\bf \tilde{u} - \tilde{v}})({\bf \tilde{u} - \tilde{v}})} 
        = \sqrt{(\frac{3}{4} - \frac{1}{2})^2 + (1 - (-2))^2} 
        = \sqrt{\frac{145}{16}} \approx 3.01 \tag{37}

    As we can see, the Pythagorean method only works in the standard Euclidean
    basis (because it's orthonormal), once we change basis we have to account
    for the distortion.


    Now back to our metric tensor, we can transform our metric tensor
    (:math:`g`) to the new basis (:math:`\tilde{g}`) using the forward "with
    basis" transform (switching to Einstein notation):

    .. math::

        \tilde{\bf g}_{ij} = T^k_l T^l_j g_{kl} =
                \begin{bmatrix} 2 & 1 \\ -\frac{1}{2} & \frac{1}{4} \end{bmatrix}
                \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
                \begin{bmatrix} 2 &  -\frac{1}{2} \\ 1 & \frac{1}{4} \end{bmatrix}
        = \begin{bmatrix} 5 & -\frac{3}{4} \\ -\frac{3}{4} & \frac{5}{16} \end{bmatrix}
        \tag{38}

    Calculating the angle and distance using Equation 32:

    .. math::

        d(\tilde{\bf u}, \tilde{\bf v})
        &= \sqrt{\tilde{g_{ij}} \tilde{u}^i \tilde{v}^j }
        = \sqrt{
                \begin{bmatrix} \frac{3}{4} & 1 \end{bmatrix}
                \begin{bmatrix} 5 & -\frac{3}{4} \\ -\frac{3}{4} & \frac{5}{16} \end{bmatrix}
                \begin{bmatrix} \frac{1}{2} \\ -2 \end{bmatrix}
            }
        = \sqrt{2} \\
        ||\tilde{\bf u}||
        &= \sqrt{\tilde{g_{ij}} \tilde{u}^i \tilde{u}^j }
        = \sqrt{
                \begin{bmatrix} \frac{3}{4} & 1 \end{bmatrix}
                \begin{bmatrix} 5 & -\frac{3}{4} \\ -\frac{3}{4} & \frac{5}{16} \end{bmatrix}
                \begin{bmatrix} \frac{3}{4} \\ 1 \end{bmatrix}
            }
        = \sqrt{2} \\
        ||\tilde{\bf v}||
        &= \sqrt{\tilde{g_{ij}} \tilde{v}^i \tilde{v}^j }
        = \sqrt{
                \begin{bmatrix} \frac{1}{2} & -2 \end{bmatrix}
                \begin{bmatrix} 5 & -\frac{3}{4} \\ -\frac{3}{4} & \frac{5}{16} \end{bmatrix}
                \begin{bmatrix} \frac{1}{2} \\ -2 \end{bmatrix}
            }
        = 2 \\
        cos(\theta) &= \frac{\tilde{g_{ij}} \tilde{u}^i \tilde{v}^j}{||\tilde{\bf u}||||\tilde{\bf v}||}
        = \frac{2}{(\sqrt{2})(2)} = \frac{1}{\sqrt{2}} \\
        \theta &= 45^{\circ} \\
        \tag{39}

    which line up with the calculations we did in our original basis.

We'll see the metric tensor come up again in a more general settings below.
Again, I'd encourage you to check out the videos in [2].  He's got a dozen or
so videos with some great derivations and intuition on the subject.  It goes 
in a bit more depth than me but still easily understandable. 

|h3| 1.7 Summary: A Tensor is a Tensor |h3e|

So, let's review a bit about tensors:

* A **tensor** is an object that is *invariant* under a change of basis,
  and whose coordinates change in a *special, predictable* way when changing a basis.
* A tensor can have **contravariant** and **covariant** components corresponding
  to the components of the tensor transforming *against* or *with* the change of basis.
* The **rank** (or degree or order) of a tensor is the number of "axis" or
  components it has (not to be confused with the dimension of each "axis").
* A :math:`(n, m)`-tensor has :math:`n` contravariant components and :math:`m`
  covariant components with rank :math:`n+m`.


We've looked at four different types of tensors:

.. csv-table::
   :header: "Tensor", "Type", "Example"
   :widths: 15, 5, 15

   "Contravariant Vectors (vectors)", "(1, 0)", "Geometric (Euclidean) vectors"
   "Covariant Vectors", "(0, 1)", "Linear Functionals"
   "Linear map", "(1,1)", "Linear Transformations"
   "Bilinear form", "(0, 2)", "Metric Tensor"


|h2| 2. Metric Space |h2e|

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


|h2| 3. Manifolds |h2e|

Now we're getting into something interesting: manifolds!  The first place
most ML hear about it is in the 
`manifold hypothesis <https://www.quora.com/What-is-the-Manifold-Hypothesis-in-Deep-Learning>`__:

    The manifold hypothesis is that real-world high dimensional data (such as
    images) lie on low-dimensional manifolds embedded in the high-dimensional
    space.

The main idea here is that even though our real-world data is high-dimensional,
there is actually some lower-dimensional representation.  For example, all "cat
images" might lie on a lower-dimensional manifold compared to say their
original 256x256x3 image dimensions.  Intuitively, makes sense that this
lower dimensional representation might be more easily learnable than an
arbitrary 256x256x3 function.

Okay, that's all well and good, but *what is a manifold?*  The abstract
definition from topology is well... abstract.  So I won't go into all the
technical details (also because I'm not super qualified to do so), but we'll
see that the Riemannian manifold is surprisingly intuitive once you get the
hang (or should I say twist) of things.

|h3| 3.1 Circles and Spheres as Manifolds |h3e|

A **manifold** is topological space that "locally" resembles Euclidean space.
This obviously doesn't mean much unless you've studied topology.  
An intuitive (but not exactly correct) way to think about it is taking
a geometric object from :math:`\mathbb{R}^k` and trying to "fit" it into
:math:`\mathbb{R}^n, n>k`.  Let's take a first example, a line segment, which
is obviously one dimensional. 

One way to embed a line in two dimensions is to "wrap" it around into a circle,
shown in Figure 2.  Each arc of the circle locally looks closer to a line
segment, and if you take an infinitesimal arc, it will "locally" resemble a
one dimensional line segment.  

.. figure:: /images/manifold_circle.png
  :height: 250px
  :alt: A Circle is a Manifold
  :align: center

  Figure 2: A circle is a manifold in two dimensions where each arc of the
  circle is locally resembles a line (source: Wikipedia).

Of course, there is a much more 
`precise definition <https://en.wikipedia.org/wiki/Topological_manifold#Formal_definition>`__ 
from topology in which a manifold is defined as a set that is 
`homeomorphic <https://en.wikipedia.org/wiki/Homeomorphism>`__ 
to a Euclidean space.  A homeomorphism is a special kind of continuous one-to-one 
mapping that preserves topological properties.  The definition is quite
abstract because the definition says a manifold is just a special kind of set
without any explicit reference of how it can be viewed as a geometric object.
I'll leave it to you if you want to go deeper into the abstract formalism of
topology, but for now let's keep it at a much more intuitive level.

Actually any "closed" loop in one dimension is a manifold because you can
imagine "wrapping" it around into the right shape.  Another way to think about
it (from the formal definition) is that from a line (segment), you can find a
continuous one-to-one mapping to a closed loop.  An interesting point is that
figure "8" is not a manifold because the crossing point does not locally
resemble a line segment.

These closed loop manifolds are the easiest 1D manifolds to think about but
there are other weird cases too shown in Figure 3.  As you can see, we can have
a variety of different shapes.  The big idea is that we can also have "open
ended" curves that extend out to infinity, which are natural mappings from
a one dimensional line.

.. figure:: /images/manifold_1d_other.png
  :height: 250px
  :alt: Other 1D Manifolds
  :align: center

  Figure 3: Circles, parabolas, hyperbolas and 
  `cubic curves <https://en.wikipedia.org/wiki/Cubic_curve>`__ 
  are all 1D Manifolds.  Note: the four different colours are all on separate
  axes and extend out to infinity if it has an open end (source: Wikipedia).

Let's now move onto 2D manifolds. The simplest one is a sphere.  You can
imagine each infinitesimal patch of the sphere locally resembles a 2D Euclidean
plane.  Similarly, any 2D surface (including a plane) that doesn't
self-intersect is also a 2D manifold.  Figure 4 shows some examples.

.. figure:: /images/manifold_2d.gif
  :height: 350px
  :alt: 1D Manifolds
  :align: center

  Figure 4: Non-intersecting closed surfaces in :math:`\mathbb{R}^3` are 
  examples of 2D manifolds such as a sphere, torus, double torus, cross
  surfaces and Klein bottle (source: Wolfram).

For these examples, you can imagine that at each point on these manifolds it
locally resembles a 2D plane.  This best analogy is Earth.  We know that the
Earth is round but when we stand in a field it looks flat.  We can of course
have higher dimension manifolds embedded in even larger dimension Euclidean
spaces but you can't really visualize them.  Abstract math is rarely easy to
visualize in higher dimension.

Hopefully after seeing all these examples, you've developed some intuition
around manifolds.  In the next section, we'll head back to the math with some
differential geometry.

|h3| 3.2 "Smooth" Riemannian Manifolds |h3e|

The examples in the last section all had some nice properties: they were
"smooth" (and not just because they are good talkers)!  We want to turn our
investigation of manifolds to "well-behaved" ones where we can do all the nice
calculus-related operations such as differentiation or integration in order to
do nice things like calculate distance or area/volume.  

The types of manifolds we want to study are called `smooth manifolds
<https://en.wikipedia.org/wiki/Differentiable_manifold#Definition>`__.  The
actual definition involves going deep into the topological definitions but the
main idea is that the manifold is that each "patch" of Euclidean space on the
manifold transitions to adjacent "patches" in a smooth way.  This allows us
to do nice calculus-like things analogous to `smooth functions
<https://en.wikipedia.org/wiki/Smoothness>`__.

To actually calculate things like distance on a manifold, we have to 
introduce a few concepts.  The first is a **tangent space** :math:`T_x M`
of a manifold :math:`M` at a point :math:`x`.  It's pretty much exactly
as it sounds: imagine you are passing through the point :math:`x` on a smooth
manifold, as you pass though you implicitly have a **tangent vector** along the
direction of travel, which can be thought of as your velocity through point :math:`x`.
The tangent vectors made in this way from each possible path passing through
:math:`x` make up the tangent space.  In two dimensions, this would be a plan.
Figure 5 shows an example of this on a sphere.

.. figure:: /images/tangent_space.png
  :height: 250px
  :alt: Tangent Space
  :align: center

  Figure 5: A tangent space at a point on a 2D manifold (a sphere) (source:
  Wikipedia).

In more detail, let's define the curve as for a manifold embedded in
:math:`\mathbb{R}^n` space with start and end points :math:`t \in [a, b]` as:

.. math::

    {\gamma}(t) := [x^1(t), \ldots, x^n(t)] \tag{40}

where :math:`x^i(t)` are single output functions of :math:`t` for component
:math:`i` (not exponents.  In this case the `tangent vector
<https://en.wikipedia.org/wiki/Tangent_vector>`__ :math:`\bf v` at :math:`x` is
just given by the derivative of :math:`\gamma(t)` with respect to :math:`t`:

.. math::

    {\bf v} := \frac{d\gamma(t)}{dt}\Big|_{t=t_x} = \Big[\frac{dx^1(t_x)}{dt}\Big|_{t=t_x}, \ldots, \frac{x^n(t)}{dt}\Big|_{t=t}\Big] \tag{41}
    
where :math:`\gamma(t_x) = x`.  Figure 6 shows another visualization of this
idea with curve :math:`{\bf \gamma}(t)` on the manifold :math:`M`. 

.. figure:: /images/tangent_space_vector.png
  :height: 250px
  :alt: Tangent Vector
  :align: center

  Figure 6: A tangent space :math:`T_x M` for manifold :math:`M` with tangent
  vector :math:`{\bf v} \in T_x M`, along a curve travelling through :math:`x \in M`
  (source: Wikipedia).


You'll be happy to know that tangent vectors are actually *contravariant*
(we didn't waste all that time talking about tensors for nothing)!  We
can easily show that for a given change of coordinates (potentially non-linear
but one-to-one)
given by the function :math:`T`
to :math:`u^i = T^i(x^1, \ldots, x^n), 1\leq i\leq n`.
Starting with the tangent vector :math:`\tilde{\bf v} = \tilde{v}^i` 
(switching to Einstein notation) in the :math:`u^i`-coordinate system:

.. math::

    \tilde{v}^i
    = \frac{dT}{dt} 
    = \frac{\partial T^i}{\partial x^s} \frac{dx^s}{dt}
    = v^i \frac{\partial T^i}{\partial x^s}



- **Riemannian metric tensor** 
- **Riemannian manifold** 

|h3| 3.3 Computing Arc Length |h3e|

- Defines arc length, thus distance using differential geometry, inner product
- Explain how to change coordinates 
- Example of computing arc length of a circle with the metric tensor,
  show it in another basis (polar coordinates?)
- Example for unit sphere???

|h2| 4. Conclusion |h2e|



|h2| 5. Further Reading |h2e|

* Wikipedia: `Tensors <https://en.wikipedia.org/wiki/Tensor_(disambiguation)>`__,
  `Manifold <https://en.wikipedia.org/wiki/Manifold>`__,
  `Metric Tensor <https://en.wikipedia.org/wiki/Metric_tensor>`__,
  `Metric Space <https://en.wikipedia.org/wiki/Metric_space>`__,
  `Covariance and contravariance of vectors <https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors>`__,
  `Vector <https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics)>`__
* [1] `Tensors for Laypeople <http://www.markushanke.net/tensors-for-laypeople/>`__, Markus Hanke
* [2] `Tensors for Beginners (YouTube playlist) <https://www.youtube.com/playlist?list=PLJHszsWbB6hrkmmq57lX8BV-o-YIOFsiG>`__, eigenchris
* `An Introduction for Tensors for Students of Physics and Engineering <https://www.grc.nasa.gov/www/k-12/Numbers/Math/documents/Tensors_TM2002211716.pdf>`__

|h2| Appendix A: Showing a Bilinear is a (0,2)-Tensor using Matrix Notation |h2e|

Let :math:`B` be our bilinear, :math:`u, v` geometric vectors,
:math:`R` our basis transform, and :math:`\tilde{B}, \tilde{u}, \tilde{v}` our
post-transformed bilinear and vectors, respectively.  Here's how we can show
that the bilinear transforms like a (0,2)-tensor using matrix notation:

.. math::

    \tilde{\bf u}^T \tilde{B} \tilde{\bf v} &= {\bf u}^T B {\bf v} && \text{output scalar same in any basis} \\
    &= (R\tilde{\bf u})^T B R\tilde{\bf v} && {\bf u}=R\tilde{\bf u}\\
    &= \tilde{\bf u}^TR^T B R\tilde{\bf v}  \\
    &= \tilde{\bf u}^T (R^T B R)\tilde{\bf v}  \\
    \therefore \tilde{B} & = R^T B R \\
   \tag{A.1} 

Note: that we can only write out the matrix representation because we're still
using rank 2 tensors. When working with higher order tensors, we can't fall back
on our linear algebra anymore.


.. [1] This is not exactly the best example because it's showing a vector in both contravariant and tangent covector space, which is not exactly the point I'm trying to make here.  But the idea is basically the same: the vector is the same object regardless of what basis you use.

.. [2] There is a `geometric interpretation of covectors <https://en.wikipedia.org/wiki/Linear_form#Visualizing_linear_functionals>`__ are parallel surfaces and the contravariant vectors "piercing" these surfaces.  I don't really like this interpretation because it's kind of artificial and doesn't have any physical analogue that I can think of.



