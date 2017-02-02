.. title: The Calculus of Variations
.. slug: the-calculus-of-variations
.. date: 2017-01-30 08:08:38 UTC-05:00
.. tags: variational calculus, mathjax
.. category: 
.. link: 
.. description: A primer on variational calculus.
.. type:

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


This post is going to describe a specialized type of calculus called
variational calculus, which helps us find the *function* that minimizes or
maximizes a given condition.
Analogous to the usual methods of calculus that we learn in university,
this one deals with functions *of functions* called *functionals* and how to
minimize or maximize them.  It's used extensively in physics such as finding
the minimum energy path a particle takes under certain conditions.  As you can
also imagine, it's also used in machine learning/statistics where you want to
find the density that optimizes a constraint [1]_.  The explanation I'm going
to use is heavily based upon Svetitsky's *notes of functionals*, which so far
is the most intuitive explanation I have read.  I'll try to follow Svetitsky's
notes to give some intuition of how we arrive at variational calculus from
regular calculus followed by a few examples so that we can get the hang of it.
With the right intuition and explanation, it's actually not too difficult,
enjoy!

.. TEASER_END

|h2| From multivariable functions to functionals |h2e|

Consider a regular scalar function :math:`F(y)`, it maps a single value to a
single value.  
You can differentiate it to get :math:`\frac{dF}{dy} = F'(y)`.
Another way to think about it is starting at :math:`y_0`, move a tiny
step away, call it :math:`dy`, then :math:`F` will change by: 
:math:`dF = F(y_0 + dy) - F(y_0) = F'(y)|_{y_0} dy`.

Now let's consider the same situation with a multivariable function
:math:`F(y_1, y_2, \ldots, y_n)`.  It maps a set of values :math:`y_1, \ldots,
y_n` to a single value :math:`F(y_1, y_2, \ldots, y_n)`.
You can also differentiate it by taking partial derivatives:
:math:`\frac{dF}{dy_1}, \ldots, \frac{dF}{dy_n}`.  Similarly, if I move
a tiny step away in each direction :math:`dy_1, \ldots, dy_n`, starting from
points :math:`y_1^0, \ldots, y_n^0`, my function will move by:

.. math::

    dF = \frac{dF}{dy_1}\Big|_{y_1^0} + \ldots + \frac{dF}{dy_n}\Big|_{y_n^0} \tag{1}

Instead of just calling the variables :math:`y_1, \ldots, y_n`, let's name them
collectively as :math:`y_n`


|h2| Further Reading |h2e|

* Wikipedia: `Calculus of Variations <https://en.wikipedia.org/wiki/Calculus_of_variations>`_
* "`Notes on Functionals <http://julian.tau.ac.il/bqs/functionals/functionals.html>`_", B. Svetitsky

.. [1] As you have probably guessed, this is the primary reason I'm interested in this area of mathematics.  A lot of popular ML/statistics techniques have the word "variational", which they get becasue they are somehow related to variational calculus.
