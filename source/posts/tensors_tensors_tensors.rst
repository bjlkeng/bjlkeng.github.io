.. title: Tensors, Tensors, Tensors
.. slug: tensors-tensors-tensors
.. date: 2018-03-13 08:24:57 UTC-05:00
.. tags: tensors, metric tensor, bilinear, linear transformations, geometric vectors, covectors, covariance, contravariance, mathjax
.. category: 
.. link: 
.. description: A quick introduction to tensors for the uninitiated.
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
In particular, tensors.  This is a topic that is casually mentioned in machine
learning papers but for those of us who weren't physics or math majors 
(\*cough\* computer engineers), it's a bit murky trying to understand what's going on.  
So on my most recent vacation, I started reading a variety of sources on the
interweb trying to piece together a picture of what tensors were all
about.  As usual, I'll skip the heavy formalities (partly because I probably
couldn't do them justice) and instead try to explain the intuition using my
usual approach of examples and more basic maths.  I'll sprinkle in a bunch of
examples and also try to relate it back to ML where possible.  Hope you like
it!

.. TEASER_END


|h2| A Tensor by Any Other Name |h2e|

For newcomers to ML, the term "tensor" has to be one of the top ten confusing
terms.  Not only because the term is new, but also because it's used
ambiguously with other branches of mathematics and physics!  In ML, it's
colloquially used interchangeably with a multidimensional array.  That's
what people usually mean when they talk about "tensors" in the context of
things like TensorFlow.  However, tensors as multidimensional arrays is just
one very narrow "view" of a tensor, tensors (mathematically speaking) are much
more than that!  Let's start at the beginning.

(By the way, you should checkout [1], which is a great series of videos
explaining tensors from the beginning.  It definitely helped clarify a lot of
ideas for me and a lot of this post is based on his presentation.)

|h2| Geometric Vectors as Tensors |h2e|

We'll start with a concept we're all familiar with: 
`geometric vectors <https://en.wikipedia.org/wiki/Euclidean_vector>`__ 
(also called Euclidean vectors).
Now there are many different variants of "vectors" but we want to talk specifically
about the geometric vectors that have a magnitude and direction.  
In particular, we're *not* talking about just an ordered pair of numbers 
(e.g.  :math:`[1, 2]` in 2 dimensions).

Of course, we're all familiar with representing geometric vectors as ordered
pairs but that's probably because we're just *assuming* that we're working in
Euclidean space where each of the indices represent the component of the basis
vectors (e.g. :math:`[1, 0]` and :math:`[0, 1]` in 2 dimensions).  If we change
basis to some other 
`linearly independent <https://en.wikipedia.org/wiki/Linear_independence>`__
basis, the components will change, but will the magnitude and direction change?
*No!*  It's still the same old vector with the same magnitude and direction.
When changing basis, we're just describing or "viewing" the vector in a
different way but fundamentally it's still the same old vector.  Figure 1 shows
a visualization.
 

.. figure:: /images/vector_tensor.png
  :height: 250px
  :alt: A Physical Vector
  :align: center

  Figure 1: The geometric vector A (in red) is the same regardless of what basis
  you use (source: Wikipedia).

You can see in Figure 1 that we have a vector :math:`A` (in red) that can
be represented in two different bases: :math:`e^1, e^2` (blue) and :math:`e_1,
e_2` (yellow) [1]_.  You can see it's the same old vector, it's just that the
way we're describing it has changed.  In the former case, we can describe it 
as a `coordinate vector <https://en.wikipedia.org/wiki/Coordinate_vector>`__
by :math:`[a_1, a_2]`,
while in the latter by the coordinate vector :math:`[a^1, a^2]` (note: the
super/subscripts represent different values, not exponents, which we'll get to
later, and you can ignore all the other stuff in the diagram).

So then a geometric vector is the geometric object, *not* specifically its
representation in a particular basis.

.. admonition:: Example 1: A geometric vector in a different basis.

    Let's take the vector :math:`v` as :math:`[a, b]` in the standard Euclidean
    basis: :math:`[1, 0]` and :math:`[0, 1]`.  Another way to write this is as:

    .. math::

        v = a \begin{bmatrix} 1 \\ 0 \end{bmatrix}
          + b \begin{bmatrix} 0 \\ 1 \end{bmatrix}
        \tag{1}

    Now what happens if we `scale <https://en.wikipedia.org/wiki/Scaling_(geometry)>`__
    our basis by :math:`2`?  This can be represented by multiplying our basis matrix
    (where each column is one of our basis vectors) by a transformation matrix:

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
    new basis. We can see that this results in the exact same vector regardless of
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
        = \begin{bmatrix} cos(\frac{\pi}{4}) & -sin(\frac{\pi}{4}) \\ 
                        sin(\frac{\pi}{4}) & cos(\frac{\pi}{4}) \end{bmatrix}
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
these geometric vectors are tensors!  Since it has one physical axis,
it is said to be a *rank=1* tensor.  A scalar is said to be a *rank=0* tensor,
which is pretty much just a degenerate case.  Note: rank is different
than dimension.

In physics and other domains, you may want to work in a non-standard Euclidean
basis because it's more convenient, but still want to talk about the same
objects regardless if we're in a standard basis or not.

So geometric vectors are our first step in understanding tensors.  To summarize
some of the main points:

* Tensors can be viewed as an ordered list of numbers with respect to a basis
  but that isn't the tensor itself.
* They are independent of a change in basis (i.e. their representation changes
  but what they represent does not).
* The *rank* (or *degree* or *order*) of a tensor specifies how many axes you
  need to specify it (careful this is different than the dimensional space
  we're working in).

Just to drive the first point home, Example 2 shows an example of a tuple
that might look like it represents a tensor but does not.

.. admonition:: Example 2: Non-Tensors

    We can represent the height, width and length of a box as an ordered list
    of numbers: :math:`[10, 20, 15]`.  However, this is not a tensor because if
    we change our basis, the height, width and length of the box don't change,
    they stay the same.  Tensors, however, have specific rules of how to change
    their representation when the basis changes.  Therefore, this tuple is not
    a tensor.


|h2| Covariant vs. Contravariant Tensors |h2e|

In the last section, we saw how geometric vectors as tensors are invariant to
basis transformations and how you have to multiply the inverse of the basis
transformation matrix with the coordinates in order to maintain that invariance
(Example 1).  Well it turns out depending on the type of tensor, how you
"maintain" the invariance can mean different things.

A geometric vector is an example of a **contravariant** vector because when
changing basis, the components of the vector transform with the inverse of the
basis transformation matrix (Example 1).  It's easy to remember it as
"contrary" to the basis matrix.  As convention, we will usually label
contravariant vectors with a superscript and write them as column vectors:

.. math::
    
    v^\alpha = \begin{bmatrix} v^0 \\ v^1 \\ v^2 \end{bmatrix}  \tag{9}

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
First, they do *not* represent geometric vectors (or else they'd be
contravariant).  Instead, we should think of them as a linear function that
takes a vector as input (in a particular basis) and maps it to a scalar, i.e.:

.. math::

    f({\bf x}) = v_0 x_0 + v_1 x_1 + v_2 x_2 \tag{11}

This is an important idea: a covariant vector is an object that has an
input (vector) and produces an output (scalar), independent of the basis you
are in.  In contrast, a contravariant vector like a geometric vector, takes no
input and produces an output, which is just itself (the geometric vector).
This is a common theme we'll see in tensors: input, output, and independent of
basis.  Let's take a look at an example of how covectors arise.


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
        g(x, y) &:= 2x + 4y  && \text{rename vars}\\ \tag{13}

    where in the last equation, I just relabelled things in terms of :math:`g,
    x, \text{ and } y` respectively, which makes it look exactly like a linear
    functional!

    As we would expect with a tensor, the "behaviour" of this covector shouldn't
    really change even if we change basis.  If we evaluate this functional
    at a geometric vector :math:`v=(a, b)` in the standard Euclidean basis,
    then of course we get :math:`g(a,b)=2a + 4b`, a scalar.  If this truly is a
    tensor, this scalar should not change even if we change our basis.

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
    change, then we should just need to multiply our covector, call it
    :math:`u = [2, 4]` (as a row vector in the standard Euclidean basis) by our
    transformation matrix:

    .. math::

        u_{R} = u * R &= [2, 4] \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
            &= [3\sqrt{2}, \sqrt{2}] \\
            \tag{16}

    Evaluating :math:`v_R` at :math:`u_R`:

    .. math::

        u_R (v_R) &= 3\sqrt{2} (\frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}})
                   + \sqrt{2} (\frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}) \\
                  &= 3a + 3b - a + b \\
                  &= 2a + 4b \\ \tag{17}

    which is precisely the scalar that we got in the Euclidean basis. 

Before we move on, I want to introduce some more notation to simply our lives.
From Equation 11, using some new notation, we can re-write covector
:math:`u_\alpha` with input geometric vector :math:`v^\alpha` (specified by
their coordinates in the same basis) as:

.. math::

    <u_\alpha, v^\alpha> = \sum_{\alpha=0}^2 u_\alpha v^\alpha
    = u_0 v^0 + u_1 v^1 + u_2 v^2 = u_\alpha v^\alpha \tag{18}

Note as before the superscripts are *not* exponentials but rather denote
an index.
The last expression uses the **Einstein summation convention**: if the
same "loop variable" appears once in both a lower and upper index, it means to
implicitly sum over that variable.  This is standard notation in physics
textbooks and makes the tedious step of writing out summations much easier.
Also note that covectors have a subscript and contravariant vectors have a
superscript, which allows them to "cancel out" via summation.  This becomes
more important as we deal with higher order tensors.

One last notational point is that we now know of two types of rank 1 tensors:
contravariant vectors (e.g. geometric vectors) and covectors (or linear
functionals).  Since they're both rank 1, we need to be a bit more precise.
We'll usually write of a :math:`(n, m)`-tensor where :math:`n` is the 
number of contravariant components and :math:`m` is the number of covariant
components.  The rank is then the sum of :math:`m+n`.  Therefore a
contravariant vector is a :math:`(1, 0)`-tensor and a covector is a 
:math:`(0, 1)`-tensor.


|h2| Linear Transformations as Tensors |h2e|

Another familiar transformation that we see is a 
`linear transformation <https://en.wikipedia.org/wiki/Linear_map>`__
(also called a linear map).  Linear transformations are just
like we remember from linear algebra, basically matrices.
*But* a linear transformation is still the same linear transformation
when we change basis so it is also a tensor (with a matrix view being one view).

Let's review a linear transformation:

    A function :math:`L:{\bf u} \rightarrow {\bf v}` is a linear map if for any
    two vectors :math:`\bf u, v` and any scalar `c`, the following two
    conditions are satisfied (linearity):

    .. math::
        L({\bf u} + {\bf v}) &= L({\bf u}) + L({\bf v}) \\
        L(c{\bf u}) &= cL({\bf u})
        \tag{19}

One key idea here is that a linear transformation takes a vector :math:`\bf v`
to another vector :math:`L(\bf v)` *in the same basis*.  The linear transformation
itself has nothing to do with the basis (we of course can apply it to a basis too).
Even though the "output" is a vector, it's analogous to the tensors we saw
above: an object that acts on a vector and returns something, independent of
the basis.

Okay, so what kind of tensor is this?  Let's try to derive it!
Let's suppose we have a geometric vector :math:`\bf v` and its transformed
output :math:`{\bf w} = L{\bf v}` in an original basis, where :math:`L` is our linear
transformation (we'll use matrix notation here).
After some change in basis via a transform :math:`T`,
we'll end up with the same vector in the new basis :math:`\bf \tilde{v}` 
and the corresponding transformed version :math:`\tilde{\bf w} = \tilde{L}{\bf \tilde{v}}`.
Note that since we're in a new basis, we have to use a new view of :math:`L`,
which we label as :math:`\tilde{L}`.

.. math::

    \tilde{L}{\bf \tilde{v}} &= \tilde{\bf w} \\
    &= T^{-1}{\bf w}  && {\bf w}\text{ is contravariant} \\ 
    &= T^{-1}L{\bf v}  && \text{definition of }{\bf w} \\ 
    &= T^{-1}LT\tilde{\bf v}  && \text{since } {\bf v} = T\tilde{\bf v} \\ 
    \therefore \tilde{L}& = T^{-1}LT \\
    \tag{20}

The second last line comes from the fact that we're going from the new basis to the old
basis so we use the inverse of the inverse -- the original basis transform.

Equation 20 tells us something interesting, we're not just multiplying by the 
inverse transform (contravariant), nor just the forward transform (covariant),
we're doing both, which hints that this is a (1,1)-tensor!  Indeed, this is
our first example of a rank 2 tensor, which usually is represented as a matrix
(e.g. 2 axes).


.. admonition:: Example 4: A Linear Transformation as a (1,1)-Tensor

    Let's start with a simple linear transformation in our standard
    Euclidean basis:

    .. math::

        L = \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & 2 \end{bmatrix}
        \tag{21}

    Next, let's use the same 45 degree rotation for our basis as Example 1 and 2
    (which also happens to be a linear transformation):

    .. math::

        R := \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                          \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix},
        \text{ }
        R^{-1} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
        \tag{22}

    Suppose we're applying :math:`L` to a vector :math:`{\bf v}=(a, b)`, and
    then changing it into our new basis.  Recall, we would first apply
    :math:`L`, then apply a contravariant (inverse matrix) transform to get to
    our new basis:

    .. math::

        R^{-1}(L{\bf v}) &= \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix}
        \Big(\begin{bmatrix} \frac{1}{2} & 0 \\ 0 & 2 \end{bmatrix}
         \begin{bmatrix} a \\ b \end{bmatrix}\Big) \\
        &=\begin{bmatrix} \frac{a}{2\sqrt{2}} + \sqrt{2}b \\ -\frac{a}{2\sqrt{2}} + \sqrt{2}b \end{bmatrix}
        \tag{23}
        
    Equation 7 tells us what :math:`\tilde{\bf v} = R^{-1}{\bf v}` is in our new basis:

    .. math:: 
        \tilde{\bf v} = \begin{bmatrix} \frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}} \\ 
                        \frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}  \end{bmatrix}  \tag{24}

    Applying Equation 20 to :math:`L` gives us:

    .. math::

        \tilde{L} &= R^{-1}LR \\ 
        &= 
        \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
                        \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} 
        \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & 2 \end{bmatrix}
        \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
                        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  \end{bmatrix} \\
        &= \begin{bmatrix} \frac{5}{4} & \frac{3}{4} \\ 
                        \frac{3}{4} & \frac{5}{4}  \end{bmatrix}\\
        \tag{25}

    Applying :math:`\tilde{L}` to  :math:`\tilde{\bf v}`:

    .. math::
        \tilde{L}\tilde{\bf v} &=
        \begin{bmatrix} \frac{5}{4} & \frac{3}{4} \\ 
                        \frac{3}{4} & \frac{5}{4}  \end{bmatrix}
        \begin{bmatrix} \frac{a}{\sqrt{2}} + \frac{b}{\sqrt{2}} \\ 
                        \frac{-a}{\sqrt{2}} + \frac{b}{\sqrt{2}}  \end{bmatrix} \\
        &= \begin{bmatrix} \frac{5a}{4\sqrt{2}} + \frac{5b}{4\sqrt{2}} 
                          - \frac{3a}{4\sqrt{2}} + \frac{3b}{4\sqrt{2}} \\
                            \frac{3a}{4\sqrt{2}} + \frac{3b}{4\sqrt{2}} 
                          - \frac{5a}{4\sqrt{2}} + \frac{5b}{4\sqrt{2}} 
         \end{bmatrix} \\
        &=\begin{bmatrix} \frac{a}{2\sqrt{2}} + \sqrt{2}b \\ -\frac{a}{2\sqrt{2}} + \sqrt{2}b \end{bmatrix} \\
        \tag{26}

    which we can see is the same as Equation 23.


|h2| Bilinear Forms |h2e|

We'll start off by introducing a not-so-familiar idea (at least by name)
called the `bilinear form <https://en.wikipedia.org/wiki/Bilinear_form>`__.
Let's take a look at the definition with respect to vector spaces:

    A function :math:`B:{\bf u, v} \rightarrow \mathbb{R}` is a bilinear form for two
    input vectors :math:`\bf u, v`, if for any other vector :math:`\bf w` and
    scalar :math:`\lambda`, the following conditions are satisfied (linearity):

    .. math::
        B({\bf u} + {\bf w}, {\bf v}) &= B({\bf u}, {\bf v}) + B({\bf w}, {\bf v}) \\
        B(\lambda{\bf u}, {\bf v}) &= \lambda B({\bf u}, {\bf v})\\
        B({\bf u}, {\bf v} + {\bf w}) &= B({\bf u}, {\bf v}) + B({\bf u}, {\bf w}) \\
        B({\bf u}, \lambda{\bf v}) &= \lambda B({\bf u}, {\bf v})\\
        \tag{27}

All this is really saying is that we have a function that maps two geometric
vectors to the real numbers, and that it's "linear" in both its
inputs (separately, not at the same time) , hence the name "bilinear".  So
again, we see this pattern: a tensor takes some input and maps it to some
output that is independent of a change in basis.

Similar to linear transformations, we can represent bilinear forms as a matrix
:math:`A`:

.. math::

    B({\bf u}, {\bf v}) = {\bf u^T}A{\bf v} = \sum_{i,j=1}^n a_{i,j}u_i v_j = A_{i,j}u^iv^j \tag{28}

where in the last expression I'm using Einstein notation to indicate that :math:`A`
is a rank (0, 2)-tensor, and :math:`{\bf u, v}` are both (1, 0)-tensors (contravariant).

So let's see how we can show that this is actually a (0, 2)-tensor (two
covector components).  We should expect that when changing basis we'll need to
multiply by the basis transform twice ("with the basis"), along the same lines
as the linear transformation in the previous section, except with two covector
components now.
We'll use Einstein notation here, but you can check out Appendix A for the equivalent
matrix multiplication operation.

Let :math:`B` be our bilinear, :math:`\bf u, v` geometric vectors, :math:`T` our basis
transform, and :math:`\tilde{B}, \tilde{\bf u}, \tilde{\bf v}` our post-transformed
bilinear form and vectors, respectively.  Here's how we can show that the
bilinear transforms like a (0,2)-tensor:

.. math::

    \tilde{B}_{ij}\tilde{u}^i\tilde{v}^j &= B_{ij}u^iv^j && \text{output scalar same in any basis} \\
    &= B_{ij}T_k^i \tilde{u}^i T^j_l \tilde{v}^j && u^i=T^i_k \tilde{u}^k \\
    &= B_{ij}T_k^i T^j_l \tilde{u}^i \tilde{v}^j && \text{re-arrange summations}\\
    \therefore \tilde{B}_{ij}& = B_{ij}T_k^i T^j_l \\
   \tag{29} 

As you can see we transform "with" the change in basis, so we get a (0, 2)-tensor.
Einstein notation is also quite convenient (once you get used to it)!

|h2| The Metric Tensor |h2e|

Before we finish talking about tensors, I need to introduce to you one of the
most important tensors around: the 
`Metric Tensor <https://en.wikipedia.org/wiki/Metric_tensor>`__.
In fact, it's probably one of the top reasons people start to learn about tensors
(and the main motivation for this post).

The definition is a lot simpler because it's just a special kind of bilinear [3]_:

    A metric tensor at a point :math:`p` is a function :math:`g_p({\bf x}_p, {\bf y}_p)`
    which takes a pair of (tangent) vectors :math:`{\bf x}_p, {\bf y}_p` at :math:`p`
    and produces a real number such that:

    * :math:`g_p` is bilinear (see previous definition)
    * :math:`g_p` is symmetric: :math:`g_p({\bf x}_p, {\bf y}_p) = g_p({\bf y}_p, {\bf x}_p)`
    * :math:`g_p` is nondegenerate.  For every :math:`{\bf x_p} \neq 0` there exists
      :math:`{\bf y_p}` such that :math:`g_p({\bf x_p}, {\bf y_p}) \neq 0`

Don't worry so much about the "tangent" part, I'm glossing over parts of it
which aren't directly relevant to this tensor discussion.

The metric tensor is important because it helps us (among other things) define
distance and angle between two vectors in a basis independent manner.  In the
simplest case, it's exactly our good old dot product operation from standard
Euclidean space.  But of course, we want to generalize this concept a little
bit so we still have the same "operation" under a change of basis i.e. the
resultant scalar we produce should be the same.  Let's take a look.

In Euclidean space, the dot product (whose generalization is called the 
`inner product <https://en.wikipedia.org/wiki/Inner_product_space>`__) 
for two vectors :math:`{\bf u}, {\bf v}` is defined as:

.. math::

    {\bf u}\cdot{\bf v} = \sum_{i=1}^n u_i v_i \tag{30}

However, for the metric tensor :math:`g` this can we re-written as:

.. math::

    {\bf u}\cdot{\bf v} = g({\bf u},{\bf v}) = g_{ij}u^iv^j 
        = [u^0, u^1] 
        \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} 
        \begin{bmatrix} v^0 \\ v^1 \end{bmatrix} \tag{31}

where in the last expression I substituted the metric tensor in standard
Euclidean space.  That is, the metric tensor in the standard Euclidean basis is
just the identity matrix:

.. math::

    g_{ij} = I_n \tag{32}

So now that we have a dot-product-like operation, we can define our
basis-independent definition of length of a vector, distance between two
vectors and angle between two vectors:

.. math::

    ||u|| &= \sqrt{g_{ij} u^i u^j} \\
    d(u, v) &= ||u-v|| = \sqrt{g_{ij} (u-v)^i (u-v)^j} \\
    cos(\theta) &= \frac{g_{ij} u^i v^j}{||{\bf u}|| ||{\bf v}||} \\
    \tag{33}

The next example shows that the distance and angle are truly invariant between
a change in basis if we use our new metric tensor definition.

.. admonition:: Example 5: Computing Distance and Angle with the Metric Tensor

    Let's begin by defining two vectors in our standard Euclidean basis:

    .. math::

        {\bf u} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, 
        {\bf v} = \begin{bmatrix} 2 \\ 0 \end{bmatrix} \tag{34}

    Using our standard (non-metric tensor) method for computing distance and
    angle:

    .. math::

        d({\bf u}, {\bf v}) &= \sqrt{({\bf u - v})({\bf u - v})} = \sqrt{(2 - 1)^2 + (0 - 1)^2}  = \sqrt{2} \\
        cos(\theta) &= \frac{{\bf u}\cdot {\bf v}}{||{\bf u}|| ||{\bf v}||} = \frac{2(1) + 1(0)}{(\sqrt{1^2 + 1^2})(\sqrt{2^2 + 0^2})} = \frac{1}{\sqrt{2}}  \\
        \theta &= 45^{\circ}
        \tag{35}

    Now, let's try to change our basis.  To show something a bit more
    interesting than rotating the axis, let's try to change to a basis
    of :math:`[2, 1]` and :math:`[-\frac{1}{2}, \frac{1}{4}]`.  To  
    change basis (from a standard Euclidean basis), the transform we need to
    apply is:

    .. math::

        T = \begin{bmatrix} 2 & -\frac{1}{2} \\ 1 & \frac{1}{4} \end{bmatrix}, 
        T^{-1} = \begin{bmatrix} \frac{1}{4} & \frac{1}{2} \\ -1 & 2 \end{bmatrix}
        \tag{36}

    As you can see, it's just concatenating the column vectors of our new basis
    side-by-side in this case (when transforming from a standard Euclidean
    space).  With these vectors, we can transform our :math:`{\bf u}, {\bf v}`
    to the new basis vectors :math:`\tilde{\bf u}, \tilde{\bf v}` as shown:

    .. math::

        \tilde{\bf u} &= T^{-1} {\bf u} = 
                \begin{bmatrix} \frac{1}{4} & \frac{1}{2} \\ -1 & 2 \end{bmatrix}
                \begin{bmatrix} 1 \\ 1 \end{bmatrix}
            = \begin{bmatrix} \frac{3}{4} \\ 1 \end{bmatrix} \\
        \tilde{\bf v} &= T^{-1} {\bf v} = 
                \begin{bmatrix} \frac{1}{4} & \frac{1}{2} \\ -1 & 2 \end{bmatrix}
                \begin{bmatrix} 2 \\ 0 \end{bmatrix}
            = \begin{bmatrix} \frac{1}{2} \\ -2 \end{bmatrix}
        \tag{37}

    Before we move on, let's see if using our standard Euclidean distance function 
    will work in this new basis:

    .. math::

        \sqrt{({\bf \tilde{u} - \tilde{v}})({\bf \tilde{u} - \tilde{v}})} 
        = \sqrt{(\frac{3}{4} - \frac{1}{2})^2 + (1 - (-2))^2} 
        = \sqrt{\frac{145}{16}} \approx 3.01 \tag{38}

    As we can see, the Pythagorean method only works in the standard Euclidean
    basis (because it's orthonormal), once we change basis we have to account
    for the distortion of the transform.

    Now back to our metric tensor, we can transform our metric tensor
    (:math:`g`) to the new basis (:math:`\tilde{g}`) using the forward "with
    basis" transform (switching to Einstein notation):

    .. math::

        \tilde{\bf g}_{ij} = T^k_i T^l_j g_{kl} =
                \begin{bmatrix} 2 & 1 \\ -\frac{1}{2} & \frac{1}{4} \end{bmatrix}
                \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
                \begin{bmatrix} 2 &  -\frac{1}{2} \\ 1 & \frac{1}{4} \end{bmatrix}
        = \begin{bmatrix} 5 & -\frac{3}{4} \\ -\frac{3}{4} & \frac{5}{16} \end{bmatrix}
        \tag{39}

    Calculating the angle and distance using Equation 33:

    .. math::

        d(\tilde{\bf u}, \tilde{\bf v})
        &= \sqrt{\tilde{g_{ij}} (\tilde{\bf u} - \tilde{\bf v})^i(\tilde{\bf u} - \tilde{\bf v})^j }
        = \sqrt{
                \begin{bmatrix} \frac{1}{4} & 3 \end{bmatrix}
                \begin{bmatrix} 5 & -\frac{3}{4} \\ -\frac{3}{4} & \frac{5}{16} \end{bmatrix}
                \begin{bmatrix} \frac{1}{4} \\ 3 \end{bmatrix}
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
        \tag{40}

    which line up with the calculations we did in our original basis.

The metric tensor comes up a lot in many different contexts (look out for
future posts) because it helps define what we mean by "distance" and "angle".
Again, I'd encourage you to check out the videos in [1].  He's got a dozen or
so videos with some great derivations and intuition on the subject.  It goes in
a bit more depth than this post but still in a very clear manner. 

|h2| Summary: A Tensor is a Tensor |h2e|

So, let's review a bit about tensors:

* A **tensor** is an object that takes an input tensor (or none at all in the
  case of geometric vectors) and produces an output tensor that is *invariant*
  under a change of basis, and whose coordinates change in a *special,
  predictable* way when changing basis.
* A tensor can have **contravariant** and **covariant** components corresponding
  to the components of the tensor transforming *against* or *with* the change of basis.
* The **rank** (or degree or order) of a tensor is the number of "axes" or
  components it has (not to be confused with the dimension of each "axis").
* A :math:`(n, m)`-tensor has :math:`n` contravariant components and :math:`m`
  covariant components with rank :math:`n+m`.

We've looked at four different types of tensors:

.. csv-table::
   :header: "Tensor", "Type", "Example"
   :widths: 15, 5, 15

   "Contravariant Vectors (vectors)", "(1, 0)", "Geometric (Euclidean) vectors"
   "Covariant Vectors", "(0, 1)", "Linear Functionals"
   "Linear Map", "(1, 1)", "Linear Transformations"
   "Bilinear Form", "(0, 2)", "Metric Tensor"

And that's all I have to say about tensors!  Like most things in mathematics,
the idea is actually quite intuitive but the math causes a lot of confusion, as
does its ambiguous use.  TensorFlow is such a cool name but doesn't exactly do
tensors justice.  Anyways, in the next post, I'll be continuing to diverge from
the typical ML topics and write about some adjacent math-y topics that pop up
in ML.

|h2| 5. Further Reading |h2e|

* Wikipedia: `Tensors <https://en.wikipedia.org/wiki/Tensor_(disambiguation)>`__,
  `Metric Tensor <https://en.wikipedia.org/wiki/Metric_tensor>`__,
  `Covariance and contravariance of vectors <https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors>`__,
  `Vector <https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics)>`__
* [1] `Tensors for Beginners (YouTube playlist) <https://www.youtube.com/playlist?list=PLJHszsWbB6hrkmmq57lX8BV-o-YIOFsiG>`__, eigenchris
* [2] `Tensors for Laypeople <http://www.markushanke.net/tensors-for-laypeople/>`__, Markus Hanke
* [3] `An Introduction for Tensors for Students of Physics and Engineering <https://www.grc.nasa.gov/www/k-12/Numbers/Math/documents/Tensors_TM2002211716.pdf>`__

|h2| Appendix A: Showing a Bilinear is a (0,2)-Tensor using Matrix Notation |h2e|

Let :math:`B` be our bilinear, :math:`\bf u, v` geometric vectors,
:math:`R` our basis transform, and :math:`\tilde{B}, \tilde{\bf u}, \tilde{\bf v}` our
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
on our matrix algebra anymore.


.. [1] This is not exactly the best example because it's showing a vector in both contravariant and tangent covector space, which is not exactly the point I'm trying to make here.  But the idea is basically the same: the vector is the same object regardless of what basis you use.

.. [2] There is a `geometric interpretation of covectors <https://en.wikipedia.org/wiki/Linear_form#Visualizing_linear_functionals>`__ are parallel surfaces and the contravariant vectors "piercing" these surfaces.  I don't really like this interpretation because it's kind of artificial and doesn't have any physical analogue that I can think of.

.. [3] Actually the metric tensor is usually defined more generally in terms of manifolds but I've simplified it here because I haven't quite got to that topic yet!

