.. title: The Calculus of Variations
.. slug: the-calculus-of-variations
.. date: 2017-02-24 08:08:38 UTC-05:00
.. tags: variational calculus, differentials, lagrange multipliers, entropy, probability, mathjax
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

    dF = \int_{a}^b \frac{\delta F}{\delta y(x)}\Big|_{y^0(x)} \delta y(x) dx \tag{5}

The meaning of Equation 5 is the same as Equation 4: a small change in :math:`F`
is proportional to a sum of small changes of :math:`\delta y(x)` (step size)
multiplied by the derivative :math:`\frac{\delta F}{\delta y(x)}` (slope),
where we can think of :math:`x` as a continuous index (analogous to :math:`i`).
As a result, the *functional derivative* is defined by:

.. math::

    \frac{\delta F}{\delta y(x)} \tag{6}

This is analogous to the derivative at each of the "indices" :math:`x`,
which we can think of as the `gradient <https://en.wikipedia.org/wiki/Gradient>`_
of the multivariate function :math:`F` (albeit with an infinite number of variables)
at each of the variables defined by :math:`y(x)`.

Equation 5 then becomes a
`directional derivative <https://en.wikipedia.org/wiki/Directional_derivative>`_,
where we can interpret as the rate of change of :math:`F` as we are
moving through "point" :math:`y^0(x)` in the direction of :math:`\delta y(x)`
(check out this `tutorial <http://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx>`_
on directional derivatives for a good intuitive referesher on the subject).

The above explanation gives a natural extension from gradients to functional derivatives
but we can also define it in terms of limits.  Using the analogy of directional
derivatives from above, we have the functional derivative at the multivariate
"point" :math:`y(x)` moving in the multivariate "direction" of an arbitrary
function :math:`\eta(x)` then we can formulate the limit as:

.. math::

    \lim_{\epsilon \to \infty} \frac{F[y(x) + \epsilon \eta(x)] - F[y(x)]}{\epsilon}
    = \int \frac{\delta F}{\delta y(x)} \eta(x) dx
    \tag{7}

which, if you think hard enough about, results in the same integral as Equation 5.
This also means the functional derivative is a function (natural extension from the
multivariate point analogy where the number of points is infinite within an interval).
We also define :math:`\delta y` as :math:`\epsilon \eta(x)` and call it the variation of
:math:`y`.

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

        \delta F(y, \eta) = \int \frac{\delta F}{\delta y(x)} \eta(x) dx \tag{8}

    As mentioned above the term :math:`\epsilon \eta(x)` is also called a
    variation of input :math:`y`, which is analogous to the infintesimely small
    :math:`epsilon` in regular calculus.

    The first variation and higher order variations define the respective
    functional derivatives and can be derived by taking the coefficients of the
    Taylor series expansion of the functional.  More details can be found
    here `Advanced Variational Methods In Mechanics Chapter 1: Variational
    Calculus Overview
    <http://www.colorado.edu/engineering/CAS/courses.d/AVMM.d/AVMM.Ch01.d/AVMM.Ch01.pdf>`_.


.. admonition:: Example 4: Computing a simple functional derivative

    Let's try to find the functional derivative of a simple functional:

    .. math::
            
        F[y(x)] = \int_0^1 y(x)^2 dx \tag{9}

    We can compute this by going back to Equation 5 and calculating 
    :math:`dF = F[y + \delta y] - F[y]`, start by computing :math:`F[y + \delta y]`:

    .. math::
            
        F[y + \delta y] &= \int_0^1 [y(x) + \delta y(x)]^2 dx \\
         &= \int_0^1 y(x)^2 + 2y(x)\delta y(x) + \delta y(x)^2 dx \\
         &= \int_0^1 y(x)^2 + 2y(x)\delta y(x) + \delta y(x)^2 dx \\
         &= F[y] + \int_0^1 2y(x)\delta y(x) dx + \int_0^1 \delta y(x)^2 dx
        \tag{10}
            
    From Equation 10, we know that in the limit :math:`\delta y \rightarrow 0` so
    we can drop the last term. Finally, computing :math:`dF` we have:

    .. math::

        dF &= F[y + \delta y] - F[y] \\
        &= F[y] + \int_0^1 2y(x)\delta y(x) dx  - F[y] \\
        &= \int_0^1 2y(x)\delta y(x) dx
        \tag{11}

    By inspection, we can see Equation 11 resembles Equation 5, thus our functional
    derivative is :math:`\frac{\delta F}{\delta y(x)} = 2y(x)`.

|h2| Euler-Lagrange Equation |h2e|

Now armed with the definition of a functional derivative, we can compute it 
from definition (if it exists).  However, as with regular calculus, computing
a derivative by definition can get tedious.  Fortunately, there is a result that
can help us compute the functional derivative called the Euler-Lagrange equation,
which states (roughly):

    For a given function :math:`y(x)` with a real argument :math:`x`, the
    functional:

    .. math::

        F[y] = \int_a^b L(x, y(x), y'(x)) dx \tag{12}

    has functional derivative given by:

    .. math::
    
        \frac{\delta F}{\delta y(x)} 
        = \frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} \tag{13}

You can derive this equation using the method we used above (and a few extra
tricks) but I'll leave it as an exercise :)
For simple functionals like Equation 12, this is a very handy way to compute functional
derivatives.  Let's take a look at a more complicated examples.

.. admonition:: Example 5: Use the Euler-Lagrange Equation to find the
    functional derivative of :math:`F[y(x)] = \int_0^1 x^3 e^{-y(x)} dx`

    Notice the second term in Equation 13 involves only :math:`y'`, which we
    doesn't appear in our functional so we that means it's 0.  Thus, the functional
    derivative in this case just treats :math:`y` as a variable and the usual rules
    of differentiation apply:

    .. math::

        \frac{\delta F}{\delta y(x)} = \frac{\partial L}{\partial y} 
        = \frac{\partial ( x^3 e^{-y(x)})}{\partial y}
        = -x^3 e^{-y(x)} \tag{14}

.. admonition:: Example 6: Use the Euler-Lagrange Equation to find the
    functional derivative of :math:`F[y(x)] = \int_0^1 x^2 y^3 y'^4 dx`

    Here we have to use all terms from Equation 13 and treat :math:`y`
    and :math:`y'` as "independent" variables:

    .. math::

        \frac{\delta F}{\delta y(x)} &= \frac{\partial L}{\partial y} 
                - \frac{d}{dx} \frac{\partial L}{\partial y'}\\
        &= 3x^2 y^2 y'^4 - 4\frac{d (x^2y^3y'^3)}{dx} \\
        &= 3x^2 y^2 y'^4 - 8xy^3y'^3 \tag{15}

|h2| Extrema of Functionals |h2e|

As an exercise this is interesting enough but the real application is when we 
want to minimize or maximize a functional in the same way we do with regular
calculus.  In the same way that we can find a point that maximizes a function,
we can find a function that maximizes a functional.

It turns out that it's pretty much what you would expect: if you set the
functional derivative to zero, we'll find a
`stationary point <https://en.wikipedia.org/wiki/Critical_point_(mathematics)>`_
of the functional where we possibly have a local minimum or maximum (i.e. a
necessary condition).  In other words, this is a place where the "slope" is
zero.  Let's take a look at a classic example.

.. admonition:: Example 7: Find the shortest possible curve between
    the points :math:`(a,c)` and :math:`(b,d)` for which the path length
    along the curve is defined by :math:`\ell(f) = \int_a^b \sqrt{1 + f'(x)^2} dx`

    First define our integrand function:
    
    .. math::
    
        L(x,y,y') = \sqrt{1 + f'(x)^2} \tag{16}

    where :math:`(x, y, y') = (x, f(x), f'(x))`.  Pre-computing the partial derivatives
    of :math:`L`:

    .. math::
    
        \frac{\partial L}{\partial y} = 0  \\
        \frac{\partial L}{\partial y'} = \frac{y'}{\sqrt{1 + y'^2}}  \tag{17}

    Plugging them into Equation 13, we get the differential equation: 

    .. math::

        \frac{d}{dx} \frac{f'(x)}{\sqrt{1 + f'(x)^2}} &= 0 \\
        \frac{f'(x)}{\sqrt{1 + f'(x)^2}} &= C \\
        f'(x) &= C\sqrt{1 + f'(x)^2} \\
        f'(x)^2 &= \frac{C^2}{1 - C^2} \\
        f'(x) &= \frac{C}{\sqrt{1 - C^2}} := A \\
        f(x) &= Ax + B \tag{18}
    
    where we introduce a constant :math:`C` after integrating, define
    a new constant :math:`A` to be the result of a constant expression with :math:`C`,
    and introduce a new constant :math:`B` from the second integration.
    As you would expect this is just a straight line through the given points.

    We can find the actual values of constants :math:`A` and :math:`B` by using 
    our initial conditions :math:`(a,c)` and :math:`(b,d)` since we know the function
    has to pass through our line:

    .. math::

        A = \frac{d-c}{b-a} \\
        B = \frac{ad-bc}{a-b} \tag{19}

Now that we have a method to solve the general problem of finding extrema for a
functional, we can add constraints in the mix.  As you may have guessed, we
can use the concept of 
`Lagrange multipliers <https://en.wikipedia.org/wiki/Lagrange_multiplier>`_ 
here (see my `previous post <link://slug/lagrange-multipliers>`_).

Given a functional in the form of Equation 12, we can add different types of constraints.
The simplest type of constraint we can have a functional constraint of the form:

.. math::

    G(y) = \int_a^b M(x, y, y') dx = C \tag{20}

In this case, the solution resembles the usual method for Lagrange multipliers.
We can solve this problem by building a new functional in the same vein of a
Lagrangian:

.. math::

    H(y) = \int_a^b (L(x,y,y') + \lambda M(x, y, y')) dx \tag{21}

Using the Euler-Lagrange equation, we can solve Equation 21 for a admissible
:math:`L` and :math:`\lambda`, keeping in mind we are given boundary conditions
at :math:`a` and :math:`b` as well as Equation 20 in order to solve for all the
constants.

The other type of constraint is just a constraint on the actual function 
(similar to regular Lagrange multipliers):

.. math::

    g(x, y) = 0 \tag{22}

Here, a Lagrange multiplier *function* needs to be introduced and the Lagrangian
becomes:

.. math::

    H(y) = \int_a^b (L(x,y,y') + \lambda(x) g(x,y)) dx \tag{23}

Again, we can use the Euler-Lagrange equation to solve Equation 23, except
we'll get a system of differential equations to solve (you need to take the functional
derivative with respect to both :math:`y(x)` and :math:`\lambda(x)`).

And at long last, we can finally get to solving some interesting problems in
probability!  Let's take a look at a couple of examples of finding 
`maximum entropy distributions <https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution>`_ 
under different constraints (check out my previous
`post <link://slug/maximum-entropy-distributions>`_ on the topic).

.. admonition:: Example 8: Find the continuous maximum entropy distribution
    with support :math:`[a,b]`.

    This is actually the same example as that appeared in my `post
    <link://slug/maximum-entropy-distributions>`_, but let's take another look.

.. admonition:: Example 9: Find the continuous maximum entropy distribution
    with support :math:`[-\infty,\infty]`, :math:`E[x] = \mu` and :math:`E[(x-\mu)^2] = \sigma^2`.

    Okay



|h2| Further Reading |h2e|

* Wikipedia: `Calculus of Variations <https://en.wikipedia.org/wiki/Calculus_of_variations>`_,
  `Functional Derivative <https://en.wikipedia.org/wiki/Functional_derivative>`_,
  `Directional Derivative <https://en.wikipedia.org/wiki/Directional_derivative>`_,
  `Differential of a function <https://en.wikipedia.org/wiki/Differential_of_a_function>`_
* `Directional Derivatives <http://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx>`_, Paul Dawkins, Paul's Online Math Notes.
* `What is the practical difference between a differential and a derivative? <http://math.stackexchange.com/questions/23902/what-is-the-practical-difference-between-a-differential-and-a-derivative>`_, Arturo Magidin, Math.Stack Exchange.
* "`Notes on Functionals <http://julian.tau.ac.il/bqs/functionals/functionals.html>`_", B. Svetitsky
* "Advanced Variational Methods In Mechanics", `Chapter 1: Variational Calculus Overview <http://www.colorado.edu/engineering/CAS/courses.d/AVMM.d/AVMM.Ch01.d/AVMM.Ch01.pdf>`_, University of Colorado at Boulder
* `Variational Problems <http://www.vgu.edu.vn/fileadmin/pictures/studies/master/compeng/study_subjects/modules/math/notes/chapter-06.pdf>`_, Vietnamese-German University.

.. [1] As you have probably guessed, this is the primary reason I'm interested in this area of mathematics.  A lot of popular ML/statistics techniques have the word "variational", which they get becasue they are somehow related to variational calculus.
