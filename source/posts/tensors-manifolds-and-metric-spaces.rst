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

|h3| Vectors as Tensors |h3e|

- vector as a tensor
- can be represented by a 1-d array but it's *not* the 1-d array
- (height, width, length) as an example of non-tensor
- different coordinate system, yields different 
- examples, pictures
- transformation independent, transform *against* basis transform

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

