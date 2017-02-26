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
variational calculus.  
Analogous to the usual methods of calculus that we learn in university,
this one deals with functions *of functions* and how to
minimize or maximize them.  It's used extensively in physics problems such as
finding the minimum energy path a particle takes under certain conditions.  As
you can also imagine, it's also used in machine learning/statistics where you
want to find a density that optimizes an objective [1]_.  The explanation I'm
going to use (at least for the first part) is heavily based upon Svetitsky's
`Notes on Functionals
<http://julian.tau.ac.il/bqs/functionals/functionals.html>`__, which so far is
the most intuitive explanation I've read.  I'll try to follow Svetitsky's
notes to give some intuition on how we arrive at variational calculus from
regular calculus with a bunch of examples along the way.  Eventually we'll
get to an application that relates back to probability.  I think with the right
intuition and explanation, it's actually not too difficult, enjoy!

.. TEASER_END

|h2| From multivariable functions to functionals |h2e|

Consider a regular scalar function :math:`F(y)`, it maps a single value :math:`y` to a
single value :math:`F(y)`.  
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
:math:`F(y_0, y_1, \ldots, y_n)` with independent variables :math:`y_0, y_1,
\ldots, y_n`.  If you squint hard enough, you may already start to see where
we're going with this: we can view the input to the function not just as
independent variables, but collectively as a group of variables derived
from a function :math:`y` (i.e. :math:`y_i = y(i), i \in \mathcal{N}`).  If
we have an infinite set of these variables, then :math:`F` is no longer just a
function of independent variables but a function of a *function* :math:`y(i)`.

Of course this is just some intuition and not at all precise but let's keep
going to see where this can take us.  Our independent points can be seen
as sampling some function :math:`y` on some interval :math:`[a, b]`.  Let's say
we want to sample :math:`N` points, each :math:`\epsilon` distance apart, 
i.e. :math:`N\epsilon = b - a`.  The :math:`n^{th}` point will be at 
:math:`x_n=a + n\epsilon`, so our sample point becomes
:math:`y_n = y(x_n) = y(a + n\epsilon)`.
As :math:`N \rightarrow \infty, \epsilon \rightarrow 0`, our sampled points
:math:`y_n` become an increasingly accurate representation of our original
function :math:`y(x)`.

Our original "function" :math:`F` can now be thought of as a function of a set
of variables :math:`\{y_n\}` resulting in :math:`F(\{y_n\})`.  As 
:math:`N \rightarrow \infty`, the original
function :math:`F` transforms into a function of an infinite set of variables,
or in another way, a function of a *function*.  The mapping is called a
*functional*, which we generally write with square brackets :math:`F[y]`.  So
instead of taking a fixed number of independent variables and outputting a
value, it takes an *infinite* number of variables, defined by :math:`y(x)` on
the interval :math:`[a,b]`, and outputs a value!

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
    
    * :math:`F[y=x] = 3`
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

    Notice that we're using the derivative of :math:`y(x)` instead of the
    function itself.  Solving for :math:`a=0, b=1, y=x`, we get:
   
    .. math::

         L[y=x] &= \int_0^1 \sqrt{1 + (\frac{d(x)}{dx})^2} dx \\
                &= \sqrt{2} \int_0^1 dx \\
                &= \sqrt{2}

A common form of functionals that in appear in many contexts is:

.. math::

    J[y] &= \int_a^b F(x, y(x), y'(x)) dx \\
    &\text{ for } x=[a,b], \text{   }a\leq b, \text{   }y(a)=\hat{y}_a, \text{   }y(b)=\hat{y}_b \tag{3}

Which is mostly just saying that :math:`y(x)` is well behaved over :math:`x\in [a,b]`.
In more detail, we want these conditions to be satisfied for any :math:`y(x)` we plug in:
:math:`y(x)` being a single-valued function, 
smooth so that :math:`y'(x)` exists as well as the integral defined in Equation 3, 
and the boundary conditions (:math:`x=[a,b], a\leq b, y(a)=\hat{y}_a, y(b)=\hat{y}_b`) 
are satisfied.

|h2| Functional Derivatives |h2e|

Now it's finally time to do something useful with functionals!  As with
regular calculus, whose premier application is finding minima and maxima,
we also want to be able to find the extrema of functionals.  It turns out we can define
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
is proportional to a sum of small changes in each direction :math:`\delta y(x)`
(step size) multiplied by the derivative for each direction :math:`\frac{\delta
F}{\delta y(x)}` (slope), where we can think of :math:`x` as a continuous index
(analogous to :math:`i`).  As a result, the *functional derivative* is defined
by:

.. math::

    \frac{\delta F}{\delta y(x)} \tag{6}

This is analogous to the derivative at each of the "independent variables" :math:`y(x)`,
which is the same concept as the `gradient <https://en.wikipedia.org/wiki/Gradient>`_
for multivariate functions.

Equation 5 then becomes a
`directional derivative <https://en.wikipedia.org/wiki/Directional_derivative>`_,
where we can interpret as the rate of change of :math:`F` as we are
moving through "point" :math:`y^0(x)` in the direction of :math:`\delta y(x)`
(check out this `tutorial <http://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx>`_
on directional derivatives for a good intuitive refresher on the subject).

This explanation takes us from gradients to functional derivatives
but we can also define it in terms of limits.  Using the analogy of directional
derivatives from above, if we have the functional derivative at the multivariate
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
    due to a small change in the functional's input.  It's the analogous concept
    to a `differential <https://en.wikipedia.org/wiki/Differential_of_a_function>`_ for
    regular calculus.

    We've already seen an example of a variation in Equation 5, which is the first
    variation of the functional :math:`F`:

    .. math::

        \delta F(y, \eta) = \int \frac{\delta F}{\delta y(x)} \eta(x) dx \tag{8}

    As mentioned above the term :math:`\epsilon \eta(x)` is also called a
    variation of input :math:`y`, which is analogous to the infinitesimally small
    :math:`\epsilon` in regular calculus.

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

    We can calculate this by going back to Equation 5 and computing its first variation: 
    :math:`dF = F[y + \delta y] - F[y]`. Start by computing :math:`F[y + \delta y]`:

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

Now armed with the definition of a functional derivative, we now know how to
compute it from first principles.  However, as with regular calculus,
computing a derivative by definition can get tedious.  Fortunately, there is a
result that can help us compute the functional derivative called the
Euler-Lagrange equation, which states (roughly):

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
derivatives.  Let's take a look at a couple more complicated examples.

.. admonition:: Example 5: Use the Euler-Lagrange Equation to find the
    functional derivative of :math:`F[y(x)] = \int_0^1 x^3 e^{-y(x)} dx`

    Notice the second term in Equation 13 involves only :math:`y'`, which we
    doesn't appear in our functional so that means it's 0.  Thus, the functional
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

As an exercise this is interesting enough, but the real application is when we 
want to minimize or maximize a functional.  In a similar way to how we find a
point that is an extremum of a function, we can also find a function that
is an extremum a functional.

It turns out that it's pretty much what you would expect: if we set the
functional derivative to zero, we'll find a
`stationary point <https://en.wikipedia.org/wiki/Critical_point_(mathematics)>`_
of the functional where we possibly have a local minimum or maximum (i.e. a
necessary condition for extrema, sometimes we might find a 
`saddle point <https://en.wikipedia.org/wiki/Saddle_point>`_ though).  In other
words, this is a place where the "slope" is zero.  Let's take a look at a
classic example.

.. admonition:: Example 7: Find the shortest possible curve between
    the points :math:`(a,c)` and :math:`(b,d)` for which the path length
    along the curve is defined by :math:`\ell(f) = \int_a^b \sqrt{1 + f'(x)^2} dx`

    First define our integrand functional:
    
    .. math::
    
        L(x,y,y') = \sqrt{1 + f'(x)^2} \tag{16}

    where :math:`(x, y, y') = (x, f(x), f'(x))`.  Pre-computing the partial derivatives
    of :math:`L`:

    .. math::
    
        \frac{\partial L}{\partial y} &= 0  \\
        \frac{\partial L}{\partial y'} &= \frac{f'(x)}{\sqrt{1 + f'(x)^2}}  \tag{17}

    Plugging them into Equation 13, we can simplify the resulting differential
    equation:

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
    As you would expect the shortest distance between two points is a straight line
    (but now you can prove it!).

    We can find the actual values of constants :math:`A` and :math:`B` by using 
    our initial conditions :math:`(a,c)` and :math:`(b,d)` since we know the function
    has to pass through our points (i.e. compute the slope and intercept of the line):

    .. math::

        A = \frac{d-c}{b-a} \\
        B = \frac{ad-bc}{a-b} \tag{19}

Now that we have a method to solve the general problem of finding extrema for a
functional, we can add constraints in the mix.  As you may have guessed, we
can use the concept of 
`Lagrange multipliers <https://en.wikipedia.org/wiki/Lagrange_multiplier>`__
here (see my previous post on `Lagrange Multipliers <link://slug/lagrange-multipliers>`__).

Given a functional in the form of Equation 12, we can add different types of constraints.
The simplest type of constraint we can have is a functional constraint of the form:

.. math::

    G[y] = \int_a^b M(x, y, y') dx = C \tag{20}

In this case, the solution resembles the usual method for Lagrange multipliers.
We can solve this problem by building a new functional in the same vein of a
Lagrangian:

.. math::

    H[y] = \int_a^b (L(x,y,y') - \lambda M(x, y, y')) dx \tag{21}

Using the Euler-Lagrange equation, we can solve Equation 21 for a function
:math:`y` and constant :math:`\lambda`, keeping in mind we are given boundary
conditions at :math:`a` and :math:`b` as well as Equation 20 to help us solve
for all the constants.  This method also naturally extends to multiple
constraints as you would expect.

The other type of constraint is just a constraint on the actual function 
(similar to regular Lagrange multipliers):

.. math::

    g(x, y) = 0 \tag{22}

Here, a Lagrange multiplier *function* needs to be introduced and the Lagrangian
becomes:

.. math::

    H[y] = \int_a^b (L(x,y,y') - \lambda(x) g(x,y)) dx \tag{23}

Again, we can use the Euler-Lagrange equation to solve Equation 23, except
we'll get a system of differential equations to solve (you need to take the functional
derivative with respect to both :math:`y(x)` and :math:`\lambda(x)`).

And at long last, we can finally get to solving some interesting problems in
probability!  Let's take a look at a couple of examples of finding 
`maximum entropy distributions <https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution>`__
under different constraints (check out my previous
post on `Maximum Entropy Distributions <link://slug/maximum-entropy-distributions>`_).

.. admonition:: Example 8: Find the continuous maximum entropy distribution
    with support :math:`[a,b]`.

    This is actually the same example as that appeared in my post on `Maximum Entropy Distributions <link://slug/maximum-entropy-distributions>`__, but let's take another look.

    First since we're finding the maximum entropy distribution, we define the
    `differential entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)#Differential_entropy>`_ 
    functional (we use :math:`H` here to denote entropy):

    .. math::

        H[f] := -\int_{a}^{b} f(x)\log(f(x)) dx \tag{24}

    Next, we define a functional constraint that our density must sum to 1:

    .. math::

        G[f] := \int_{a}^{b} f(x) dx = 1 \tag{25}

    Now put together the Lagrangian equivalent:

    .. math::

        F[f] &= \int_{a}^{b} L(x, f(x)) dx \\
             &= \int_{a}^{b} -f(x)\log(f(x)) - \lambda f(x) dx \tag{26}

    Using the Euler-Lagrange equation to find the maximum and noticing we have
    no derivatives of :math:`f(x)`, we get:

    .. math::

        \frac{\delta L}{\delta f(x)} = -\log(f(x)) - 1 - \lambda &= 0 \\
        -\log(f(x)) = 1 + \lambda \\
        f(x) = e^{-\lambda - 1} \tag{27}

    Plugging this into our constraint in Equation 25:

    .. math::

        G[f] = \int_{a}^{b} e^{-\lambda - 1} dx &= 1 \\
        e^{-\lambda - 1} \int_{a}^{b} dx &= 1 \\
        e^{-\lambda - 1} &= \frac{1}{b-a} \tag{28}

    Now substituting back into Equation 27, we get:

    .. math::

        f(x) = \frac{1}{b-a} \tag{29}

    This is nothing more than a uniform distribution on the interval
    :math:`[a,b]`. This means that given no other knowledge of a distribution
    (except its support), the principle of maximum entropy says we should
    assume it's a uniform distribution.
   
.. admonition:: Example 9: Find the continuous maximum entropy distribution
    with support :math:`[-\infty,\infty]`, :math:`E[x] = \mu` and :math:`E[(x-\mu)^2] = \sigma^2`.

    You may already be able to guess what kind of distribution we should end up with
    when we have the mean and variance specified, let's see if you're right.

    We'll just transform the variance constraint into the second moment to make 
    this a bit more symmetric:

    .. math::

        \int_{-\infty}^{\infty} f(x) (x-\mu)^2 dx &= \sigma^2  \\
        \int_{-\infty}^{\infty} f(x)x^2 dx - \mu^2 &= \sigma^2  \\
        \int_{-\infty}^{\infty} f(x)x^2 dx &= \sigma^2 + \mu^2 \tag{30}

    Given this objective functional and associated constraints:

    .. math::

        H[f] &:= -\int_{-\infty}^{\infty} f(x)\log(f(x)) dx \\
        G_0[f] &:= \int_{-\infty}^{\infty} f(x) dx = 1 \\
        G_1[f] &:= \int_{-\infty}^{\infty} f(x) x  dx = \mu  \\
        G_2[f] &:= \int_{-\infty}^{\infty} f(x) x^2 dx = \sigma^2 + \mu^2 \tag{31}

    we can put together the Lagrangian functional:

    .. math::

        F[f] &:= \int_{-\infty}^{\infty} L(x, f(x)) dx \\
         &= \int_{-\infty}^{\infty} -f(x)\log(f(x))
        - \lambda_0 f(x)
        - \lambda_1 f(x) x 
        - \lambda_2 f(x) x^2 dx \tag{32}

    Using the Euler-Lagrange equation again and setting it to 0:

    .. math::

        -\log(f(x)) - 1 - \lambda_0 - \lambda_1 x - \lambda_2 x^2 &= 0 \\
        f(x) &= e^{-(1 + \lambda_0 + \lambda_1 x + \lambda_2 x^2)} \tag{33}

    Now this is not going to work out so nicely in terms of plugging it back into
    our constraints from Equation 30 because integrals involving :math:`e^{x^2}`
    usually don't have nice anti-derivatives.  But one thing to notice is that this is
    basically the form of a 
    `Gaussian function <https://en.wikipedia.org/wiki/Gaussian_function>`_
    (you'll have to do some legwork to complete the square though):

    .. math::

        f(x) &= e^{-(1 + \lambda_0 + \lambda_1 x + \lambda_2 x^2)} \\
             &= ae^{-\frac{(x-b)^2}{2c^2}} \tag{34}

    Further, we know from the constraints in Equation 31 that the function
    is normalized to :math:`1`, making this a normal distribution.
    Thus we can determine the values of the missing coefficients by just
    matching them against the definition of a normal distribution:

    .. math::

        a &= \frac{1}{\sqrt{2\pi \sigma^2}} \\
        b &= \mu \\
        c &= \sigma^2  \tag{35}

    So by the principle of maximum entropy, if we only know the mean and variance
    of a distribution with support along the real line, we should assume the distribution
    is normal.

|h2| Conclusion |h2e|

For those of us who aren't math or physics majors (`*` *cough* `*` computer engineers),
variational calculus is an important topic that we missed out on.  Not only
does it have a myriad of applications in physical domains (it's the most common
type of problem when searching for "variational calculus"), it also has many
applications in statistics and machine learning (you can expect a future post
using this topic!).  As with most things, once you know enough about the individual
parts (multivariable calculus, Lagrange multipliers etc.) the actual topic (variational calculus)
isn't too much of a stretch (at least when you're not trying to prove things
formally!).  I hope this post helps all the non-mathematicians and non-physicists
out there.


|h2| Further Reading |h2e|

* Previous Posts: `Lagrange Multipliers <link://slug/lagrange-multipliers>`__, `Max Entropy Distributions <link://slug/maximum-entropy-distributions>`__
* Wikipedia: `Calculus of Variations <https://en.wikipedia.org/wiki/Calculus_of_variations>`_,
  `Functional Derivative <https://en.wikipedia.org/wiki/Functional_derivative>`_,
  `Directional Derivative <https://en.wikipedia.org/wiki/Directional_derivative>`_,
  `Differential of a function <https://en.wikipedia.org/wiki/Differential_of_a_function>`_,
  `Lagrange multipliers <https://en.wikipedia.org/wiki/Lagrange_multiplier>`__
* `Directional Derivatives <http://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx>`_, Paul Dawkins, Paul's Online Math Notes.
* `What is the practical difference between a differential and a derivative? <http://math.stackexchange.com/questions/23902/what-is-the-practical-difference-between-a-differential-and-a-derivative>`_, Arturo Magidin, Math.Stack Exchange.
* "`Notes on Functionals <http://julian.tau.ac.il/bqs/functionals/functionals.html>`__", B. Svetitsky
* "Advanced Variational Methods In Mechanics", `Chapter 1: Variational Calculus Overview <http://www.colorado.edu/engineering/CAS/courses.d/AVMM.d/AVMM.Ch01.d/AVMM.Ch01.pdf>`_, University of Colorado at Boulder
* `Variational Problems <http://www.vgu.edu.vn/fileadmin/pictures/studies/master/compeng/study_subjects/modules/math/notes/chapter-06.pdf>`_, Vietnamese-German University.

.. [1] As you have probably guessed, this is the primary reason I'm interested in this area of mathematics.  A lot of popular ML/statistics techniques have the word "variational", which they get because they are somehow related to variational calculus.
