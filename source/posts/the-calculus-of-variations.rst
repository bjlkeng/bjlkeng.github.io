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
step away, call it :math:`dy`, then :math:`F` will change by its `differential
<https://en.wikipedia.org/wiki/Differential_of_a_function>`_:

.. math::

    dF = F(y_0 + dy) - F(y_0) = \frac{dF}{dy}|_{y_0} dy \tag{1}

Now let's consider the same situation with a multivariable function
:math:`F(y_1, y_2, \ldots, y_n)`.  It maps a set of values :math:`y_1, \ldots,
y_n` to a single value :math:`F(y_1, y_2, \ldots, y_n)`.
You can also differentiate it by taking partial derivatives:
:math:`\frac{dF}{dy_1}, \ldots, \frac{dF}{dy_n}`.  Similarly, if I move
a tiny step away in each direction :math:`dy_1, \ldots, dy_n`, starting from
points :math:`y_1^0, \ldots, y_n^0`, the 
`total differential <https://en.wikipedia.org/wiki/Differential_of_a_function#Differentials_in_several_variables>`_ 
of the function is: 

.. math::

    dF = \frac{dF}{dy_1}\Big|_{y_1^0} dy_1 + \ldots + \frac{dF}{dy_n}\Big|_{y_n^0}dy_n \tag{2}

So far we have only touched upon derivatives and differentials of a function
:math:`F(y_0, y_1, \ldots, y_n)` within independent variables :math:`y_0, y_1,
\ldots, y_n`.  If you squint hard enough, you might start to see them
not as independent variables but collectively as a function :math:`y` that
takes an input :math:`i` i.e. :math:`y_i = y(i), i \in \mathcal{N}`.  
In that case, :math:`F` is no longer just a function of independent variables
but a function of a *function* :math:`y(i)`, which is another name for a **functional**!

Of course this is just some intuition and not at all precise but let's keep
going to see if we can get to a derivative.  Our independent points can be seen
as sampling some function :math:`y` on some interval :math:`[a, b]`.  Let's say
we want to sample :math:`N` points, each :math:`\epsilon` distance apart, 
i.e. :math:`N\epsilon = b - a`.  The :math:`n^{th}` point will be at 
:math:`x=a + n\epsilon`, so our sample point becomes
:math:`y_n = y(x_n) = y(a + n\epsilon)`.

*I hope you can see where we're going with this ...*

As :math:`N \rightarrow \infty, \epsilon \rightarrow 0`, our sampled points
:math:`y_n` becomes an increasingly accurate representation of our original
function :math:`y(x)`.

Now our original "function" :math:`F` can be thought of as a function of a set
of variables :math:`\{y_n\}` resulting in :math:`F(\{y_n\})`.  As 
:math:`N \rightarrow \infty`, the original
function :math:`F` transforms into a function of a function :math:`y(x)`, also
called a *functional* which we generally write as :math:`F[y]`.  So instead
of taking a fixed number of independent variables and outputting a value,
it takes an *infinite* number of variables, defined by :math:`y(x)` on the
interval :math:`[a,b]`, and outputs a value!

Let's pause for a second and summarize: 

 * A function :math:`y(x)` takes as input a number :math:`x` and returns a number.
 * A functional :math:`F[y]` takes as input a *function* :math:`y(x)` and returns
   a number.  
   
Another way to say this is :math:`F[y(x)]` takes *all* the values of :math:`y(x)`
at all the values of :math:`x` (for that :math:`y`) and maps it to a single
number.  Let's take a look at a few examples to make things concrete.


.. admonition:: Example 1

   The simplest functional just evaluates the input function at a particular value.

   Define :math:`F[y] = y(3)`.
   
   * :math:`F[y=x] = y(3) = 3`
   * :math:`F[y=x^2] = (3)^2 = 9`
   * :math:`F[y=\ln_3(x)] = \ln_3(3) = 1`
 

.. admonition:: Example 2

   Many useful functionals will take a definite integral of the input function
   as a means to map it to a number.
   
   Define :math:`F[y] = \int_0^1  y(x) dx`.


   * :math:`F[y=x] = \int_0^1  x dx = \frac{1}{2}`
   * :math:`F[y=x^2] = \int_0^1  x^2 dx = \frac{1}{3}`
   * :math:`F[y=e^x] = \int_0^1 e^x dx = e - 1`

|h2| Functional Derivatives |h2e|




|h2| Further Reading |h2e|

* Wikipedia: `Calculus of Variations <https://en.wikipedia.org/wiki/Calculus_of_variations>`_
* "`Notes on Functionals <http://julian.tau.ac.il/bqs/functionals/functionals.html>`_", B. Svetitsky
* "Advanced Variational Methods In Mechanics", `Chapter 1: Variational Calculus Overview <http://www.colorado.edu/engineering/CAS/courses.d/AVMM.d/AVMM.Ch01.d/AVMM.Ch01.pdf>`_, University of Colorado at Boulder.

.. [1] As you have probably guessed, this is the primary reason I'm interested in this area of mathematics.  A lot of popular ML/statistics techniques have the word "variational", which they get becasue they are somehow related to variational calculus.
