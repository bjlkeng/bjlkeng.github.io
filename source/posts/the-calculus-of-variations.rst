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

    dF = \frac{\partial F}{\partial y_1}\Big|_{y_1^0} dy_1 + \ldots + \frac{\partial F}{\partial y_n}\Big|_{y_n^0}dy_n \tag{2}

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
:math:`y_n = y(x_n) = y(a + n\epsilon)`.  *(I hope you can see where we're going with this ...)*
As :math:`N \rightarrow \infty, \epsilon \rightarrow 0`, our sampled points
:math:`y_n` become an increasingly accurate representation of our original
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


.. admonition:: Example 1: A simple functional

    The simplest functional just evaluates the input function at a particular value.

    Define :math:`F[y] = y(3)`.
    
    * :math:`F[y=x] = y(3) = 3`
    * :math:`F[y=x^2] = (3)^2 = 9`
    * :math:`F[y=\ln_3(x)] = \ln_3(3) = 1`

.. admonition:: Example 2: An integral functional

    Many useful functionals will take a definite integral of the input function
    as a means to map it to a number.
    
    Define :math:`F[y] = \int_0^1  y(x) dx`.


    * :math:`F[y=x] = \int_0^1  x dx = \frac{1}{2}`
    * :math:`F[y=x^2] = \int_0^1  x^2 dx = \frac{1}{3}`
    * :math:`F[y=e^x] = \int_0^1 e^x dx = e - 1`

.. admonition:: Example 3: Functionals with derivatives

    Since we're dealing with functions as inputs, the functional can also 
    involve the derivative of an input function.
    
    A functional that defines the `arc length of a curve <http://tutorial.math.lamar.edu/Classes/CalcII/ArcLength.aspx>`_: 

    .. math::

         L[y] = \int_a^b \sqrt{1 + (\frac{dy}{dx})^2} dx

    Notice that we're using the derivtive of :math:`y(x)` instead of the
    function itself.  Solving for :math:`a=0, b=1, y=x`, we get:
   
    .. math::

         L[y=x] &= \int_0^1 \sqrt{1 + (\frac{d(x)}{dx})^2} dx \\
                &= \sqrt{2} \int_0^1 dx \\
                &= \sqrt{2}

A common form of functionals that in appear in many contexts such as physics is:

.. math::

    J[y] &= \int_a^b F(x, y(x), y'(x)) dx \\
    &\text{ for } x=[a,b], \text{   }a\leq b, \text{   }y(a)=\hat{y}_a, \text{   }y(b)=\hat{y}_b \tag{3}

Which is just mostly saying that :math:`y(x)` is well behaved over :math:`x\in [a,b]`.
In more detail, we want these conditions to be satisfied for any :math:`y(x)` we plug in:
`y(x)` being a single-valued function, 
smooth so that :math:`y'(x)` exists as well as the integral defined in Equation 3, 
and the boundary conditions (:math:`x=[a,b], a\leq b, y(a)=\hat{y}_a, y(b)=\hat{y}_b`) 
are satisfied.

|h2| Functional Derivatives |h2e|

Now it's finally time to get to do something useful with functionals!  As with
regular calculus whose premier application is finding minima and maxima,
we also want to do the same thing with functionals.  It turns out we can define
something analogous to a derivative unsurprisingly called a *functional derivative*.
Let's see how we can intuitively build it up from the same multivariable
function from above.

Let's take another look at the total differential in Equation 2 again, but re-write
it this time as a sum:

.. math::

    dF = \sum_{i=1}^N \frac{\partial F}{\partial y_i}\Big|_{y_i^0} dy_i \tag{4}

Again, if you squint hard enough, as :math:`N \rightarrow \infty`,
:math:`y_i` approximates our :math:`y(x)`, and our sum turns into an integral
of a continuous function (recall the domain of our function :math:`y(x)` was :math:`x \in [a,b]`).

.. math::

    dF = \int_{a}^b \frac{\partial F}{\partial y(x)}\Big|_{y^0(x)} \partial y(x) dx \tag{5}

The meaning of Equation 5 is the same as Equation 4: a small change in :math:`F`
is proportional to a sum of small changes of :math:`\partial y(x)` (step size)
multiplied by the derivative :math:`\frac{\partial F}{\partial y(x)}` (slope),
where we can think of :math:`x` as a continuous index (analogous to :math:`i`).
As a result, the *functional derivative* is defined by:

.. math::

    \frac{\partial F}{\partial y(x)} \tag{6}

This is analogous to the derivative at each of the "indices" :math:`x`,
which we can think of as the `gradient <https://en.wikipedia.org/wiki/Gradient>`_
of the multivariate function :math:`F` (albeit with an infinite number of variables)
at each of the variables defined by :math:`y(x)`.

Equation 5 then becomes a
`directional derivative <https://en.wikipedia.org/wiki/Directional_derivative>`_,
where we can interpret as the rate of change of :math:`F` as we are
moving through "point" :math:`y^0(x)` in the direction of :math:`\partial y(x)`
(check out this `tutorial <http://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx>`_
on directional derivatives for a good intuitive referesher on the subject).

The above explanation gives a natural extension from gradients to functional derivatives
but we can also define it in terms of limits.  Using the analogy of directional
derivatives from above, we have the functional derivative at the multivariate "point" :math:`y(x)`
moving in the multivariate "direction" of an arbitrary function :math:`\eta(x)`
then we can formulate the limit as:

.. math::

    \lim_{\epsilon \to \infty} \frac{F[y(x) + \epsilon \eta(x)] - F[y(x)]}{\epsilon}
    = \int \frac{\partial F}{\partial y(x)} \eta(x) dx
    \tag{7}

which, if you think hard enough about, results in the same integral as Equation 5.
Of course, there's no guarantee that the functional derivative exists.  That's
where formal definitions and rigorous mathematics comes in, which is beyond
the scope of this post.  Also important to mention is that we can have higher order
functional derivatives that can be defined in a very similar way.  For now,
let's just focus on simple cases where everything plays nicely.

.. admonition:: Why the name *variational* calculus?

    A variation of a functional is the small change in a functional's value
    due to a small change in the functional argument.  It's the analogous concept
    to a `differential <https://en.wikipedia.org/wiki/Differential_of_a_function>`_ for
    regular calculus.

    We've already seen an example of a variation in Equation 5, which is the first
    variation of the functional :math:`F`:

    .. math::

        \partial F(y, \eta) = \int \frac{\partial F}{\partial y(x)} \eta(x) dx \tag{8}

    The term :math:`\epsilon \eta(x)` is also called a finite variation, which is
    analogous to the infintesimely small :math:`epsilon` in regular calculus.

    The first variation and higher order variations define the respective
    functional derivatives and can be derived by taking the coefficients of the
    Taylor series expansion of the functional.  More details can be found
    here `Advanced Variational Methods In Mechanics Chapter 1: Variational
    Calculus Overview
    <http://www.colorado.edu/engineering/CAS/courses.d/AVMM.d/AVMM.Ch01.d/AVMM.Ch01.pdf>`_.


.. admonition:: Example 4: Computing a simple functional derivative

    Let's try to find the functional derivative of a simple functional:


    from the definition of 

|h2| Further Reading |h2e|

* Wikipedia: `Calculus of Variations <https://en.wikipedia.org/wiki/Calculus_of_variations>`_,
  `Functional Derivative <https://en.wikipedia.org/wiki/Functional_derivative>`_,
  `Directional Derivative <https://en.wikipedia.org/wiki/Directional_derivative>`_,
  `Differential of a function <https://en.wikipedia.org/wiki/Differential_of_a_function>`_
* `Directional Derivatives <http://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx>`_, Paul Dawkins, Paul's Online Math Notes.
* `What is the practical difference between a differential and a derivative? <http://math.stackexchange.com/questions/23902/what-is-the-practical-difference-between-a-differential-and-a-derivative>`_, Arturo Magidin, Math.Stack Exchange.
* "`Notes on Functionals <http://julian.tau.ac.il/bqs/functionals/functionals.html>`_", B. Svetitsky
* "Advanced Variational Methods In Mechanics", `Chapter 1: Variational Calculus Overview <http://www.colorado.edu/engineering/CAS/courses.d/AVMM.d/AVMM.Ch01.d/AVMM.Ch01.pdf>`_, University of Colorado at Boulder

.. [1] As you have probably guessed, this is the primary reason I'm interested in this area of mathematics.  A lot of popular ML/statistics techniques have the word "variational", which they get becasue they are somehow related to variational calculus.
