.. title: A Probabilistic View of Linear Regression
.. slug: a-probabilistic-view-of-regression
.. date: 2016-05-08 14:43:05 UTC-04:00
.. tags: regression, probability, Bayesian, mathjax
.. category: 
.. link: 
.. description: Another look at linear regression through the lens of probability.
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

One thing that I always disliked about introductory material to linear
regression is how randomness is explained.  The explanations always
seemed unintuitive because, as I have frequently seen it, they appear as an
after thought rather than the central focus of the model.  
In this post, I'm going to try to
take another approach to building an ordinary linear regression model starting
from a probabilistic point of view (which is pretty much just a Bayesian view).
After the general idea is established, I'll modify the model a bit and end up
with a Poisson regression using the exact same principles showing how
generalized linear models aren't any more complicated.  Hopefully, this will
help explain the "randomness" in linear regression in a more intuitive way.


.. TEASER_END

|h2| Background |h2e|

The basic idea behind a regression is that you want to model
the relationship between an outcome variable :math:`y` (a.k.a dependent
variable, endogenous variable, response variable), and a vector of explanatory
variables :math:`{\bf x} = (x_1, x_2, \ldots, x_n)` (a.k.a. independent variables,
exogenous variables, covariates, features, or input variables).  A 
`linear regression <https://en.wikipedia.org/wiki/Linear_regression>`_
relates :math:`y` to a linear predictor function of 
:math:`\bf{x}` (how they relate is a bit further down).  For a given data point
:math:`i`, the linear function is of the form:

.. math::

    f(i) = \beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip} \tag{1}

Notice that the function is linear in the parameters :math:`{\bf \beta} =
(\beta_0, \beta_1, \ldots, \beta_n)`, not necessarily in terms of the explanatory variables.
It's possible to use a non-linear function of another explanatory variable as an explanatory variable itself, e.g. :math:`f(i) = \beta_0 + \beta_1 x_{i} + \beta_2 x^2_{i} + \beta_3 x^3_{i}`
is a linear predictor function.

There are usually two main reasons to use a regression model:

* Predicting a future value of :math:`y` given its corresponding explanatory
  variables.  An example of this is predicting a student's test scores given
  attributes about the students.
* Quantifying the strength of the relationship of :math:`y` in terms of its
  explanatory variables.  An example of this is determining how strongly the
  unit sales of a product varies with its price (i.e. price elasticity).

The simplest form of linear regression model equates the outcome variable with
the linear predictor function (ordinary linear regression), adding an error
term (:math:`\varepsilon`) to model the noise that appears when fitting the
model.  The error term is added because :math:`y` variable almost never can be
exactly determined by :math:`{\bf x}`, there is always some noise or
uncertainty in the relationship which we want to model.

.. math::

    y_i = \beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip} + \varepsilon_i \tag{2}

From this equation, most introductory courses will go into estimating the
:math:`\beta` parameters using an `ordinary least squares
<https://en.wikipedia.org/wiki/Ordinary_least_squares>`_ approach given a set
of :math:`(y_i, {\bf x_i})` pairs, which then can be used for either prediction
or quantification of strength of the relationship.  Instead of going the
traditional route, let's start from the ground up by specifying the probability
distribution of :math:`y` and working our way back up.

|h2| Modeling the Outcome as a Normal Distribution |h2e|

Instead of starting off with both :math:`y` and :math:`\bf{x}` variables,
we'll start by describing the probability distribution of *just* :math:`y`
and *then* introducing the relationship to the explanatory variables.

|h3| A Constant Mean Model |h3e|

First, let's model :math:`y` as a standard normal distribution with a
zero (i.e. known) mean and unit variance. Note this does *not* depend any explanatory
variables (no :math:`{\bf x}`'s anywhere to be seen):

.. math::

    Y \sim N(0, 1) \tag{3}

In this model for :math:`y`, we have nothing to estimate -- all the normal
parameter distribution parameters are already set (mean :math:`\mu=0`, variance
:math:`\sigma^2=1`).
In the language of linear regression, this model would be represented as
:math:`y=0 + \varepsilon` with no dependence on any :math:`{\bf x}` values and
:math:`\varepsilon` being a standard normal distribution.  Please note that 
even though *on average* we expect :math:`y=0`, we still a expect certain amount
of fluctuation or randomness about the :math:`0`.


Next, let's make it a little bit more interesting by assuming a fixed
*unknown* mean and variance :math:`\sigma^2` corresponding to
:math:`y=\mu + \varepsilon` regression model (here :math:`\varepsilon` is a
zero mean and :math:`\sigma^2` variance):

.. math::

    Y \sim N(\mu, \sigma^2) \tag{4}
    
We are still not modeling the relationship between :math:`y` and :math:`{\bf
x}` (bear with me here, we'll get there soon).  In Equation 4, if we're given
a set of :math:`(y_i, {\bf x_i})`, we can get an unbiased estimate for
:math:`\mu` by just using the mean of all the :math:`y_i`'s
(we can also estimate :math:`\sigma^2` but let's keep it simple for now).
A more round about (but more insightful) way to find this estimate is to
maximize the `likelihood <https://en.wikipedia.org/wiki/Likelihood_function>`_
function.

|h3| Maximizing Likelihood |h3e|

Consider that we have :math:`n` points, each of which is drawn in an independent
and identically distributed (i.i.d.) way from the normal distribution in Equation 4.
For a given, :math:`\mu, \sigma^2`, the probability of those :math:`n` points
being drawn define the likelihood function, which are just the multiplication
of :math:`n` normal probability density functions (PDF) (because they are independent).

.. math::

    \mathcal{L(\mu|y)} = \prod_{i=1}^{n} P_Y(y_i|\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(y_i-\mu)^2}{2\sigma^2}} \tag{5}

Once we have a likelihood function, a good estimate of the parameters (i.e.
:math:`\mu, \sigma^2`) is to just find the combination of parameters that
maximizes this function for the given data points.  In this scenario,
the data points are fixed (we have observed :math:`n` of them with known values) and
we are trying to estimate the unknown values for :math:`\mu` (or :math:`\sigma^2`).
Here we derive the maximum likelihood estimate for :math:`\mu`:

.. math::

    \hat{\mu} = \arg\max_\mu  \mathcal{L(\mu|y)} &= \arg\max_\mu \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(y_i-\mu)^2}{2\sigma^2}} \\
    &= \arg\max_\mu \log\big(\prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(y_i-\mu)^2}{2\sigma^2}}\big)\\
    &= \arg\max_\mu \sum_{i=1}^{n} \log(\frac{1}{\sigma\sqrt{2\pi}}) + \log(e^{-\frac{(y_i-\mu)^2}{2\sigma^2}}) \\
    &= \arg\max_\mu \sum_{i=1}^{n}\log(e^{-\frac{(y_i-\mu)^2}{2\sigma^2}})  \\
    &= \arg\max_\mu \sum_{i=1}^{n} -\frac{(y_i-\mu)^2}{2\sigma^2} \\
    &= \arg\min_\mu \sum_{i=1}^{n} (y_i-\mu)^2
    \tag{6}

We use a couple of tricks here.  It turns out maximizing the likelihood is the same as
maximizing the log-likelihood [1]_ and it makes the manipulation much easier.
Also, we can remove any additive or multiplicative constants where appropriate
because they do not affect the maximum likelihood value.  

To find the actual value of the optimum point, we can take the partial
derivative of Equation 6 with respect to :math:`\mu` and setting it to zero:

.. math::

    \frac{\partial}{\partial \mu}\log\mathcal{L(\mu|y)} &= 0 \\
    \frac{\partial}{\partial \mu}\sum_{i=1}^{n} (y_i-\mu)^2  &= 0 \\
    \sum_{i=1}^{n} -2(y_i-\mu)  &= 0 \\
        n\mu = \sum_{i=1}^{n} y_i \\
        \mu = \frac{1}{n}\sum_{i=1}^{n} y_i \tag{7}

Which is precisely the mean of the :math:`y` values as expected.  Even though
we knew the answer ahead of time, this work will be useful once we complicate
the situation by introducing the explanatory variables.

Finally, the expected value of :math:`y` is just the expected value of a normal
distribution, which is just equal its mean:

.. math::

    E(y) = \mu \tag{8}

|h3| A Couple of Important Ideas |h3e|

So far we haven't done anything too interesting.  We've simply looked at how to
estimate a "regression" model :math:`y=\mu + \varepsilon`, which simply
relates the outcome variable :math:`y` to a constant :math:`\mu`.  
Another way to write this in terms of Equation 2 would be :math:`y=\beta_0 + \varepsilon`,
where we just relabel :math:`\mu=\beta_0`.

Before we move on, there are two points that I want to stress that might be easier to
appreciate with this extremely simple "regression".  First, :math:`y` is a random variable.
Assuming our model represents the data correctly, when we plot a histogram it
should bell shaped and centered at :math:`\mu`.  This is important to
understand because a common misconception with regressions is that :math:`y` is
a deterministic function of the :math:`{\bf x}` (or in this case constant) values.
This confusion probably comes about because the error term :math:`\varepsilon`
error term is tacked on at the end of Equation 2 reducing its importance.
In our constant modeling of :math:`y`, it would be silly to think of :math:`y`
to be exactly equal to :math:`\mu` -- it's not.  Rather, the values of :math:`y`
are normally distributed around :math:`\mu` with :math:`\mu` just being the
expected value.

Second, :math:`\mu = \frac{1}{n}\sum_{i=1}^{n} y_i` (from Equation 7) is a
*point estimate*.  We don't know its exact value, whatever we estimate will probably
not be equal to its "true" value (if such a thing exists).  Had we sampled our data
points slightly differently, we would get a slightly different estimate of
:math:`\mu`. *This all points to the fact that* :math:`\mu` *is a random variable*
[2]_.  I won't talk too much more about this point since it's a bit outside
scope for this post but perhaps I'll discuss it in the future.

|h2| Modeling Explanatory Variables |h2e|

Now that we have an understanding that :math:`y` is a random variable, let's
add in some explanatory variables.  We can model the expected value of
:math:`y` as a linear function of :math:`p` explanatory variables [3]_ similar to
Equation 2:

.. math::

    E(y|{\bf x}) = \beta_0 + \beta_1 x_{1} + ... + \beta_p x_{p} \tag{9}

Combining this Equation 8, the mean of :math:`y` is now just this linear
function.  Thus, :math:`y` is a normal variable with mean as a linear function
of :math:`{\bf x}` and a fixed standard deviation:

.. math::

    y \sim N(\beta_0 + \beta_1 x_{1} + ... + \beta_p x_{p}, \sigma^2) \tag{10}

This notation makes it clear that :math:`y` is still a random normal variable
with an expected value corresponding to the linear function of :math:`{\bf x}`.
The problem now is trying to find estimates for the :math:`p` :math:`\beta_i`
parameters instead of just a single :math:`\mu` value.

|h3| Maximizing Likelihood |h3e|

To get point estimates for the :math:`\beta_i` parameters, we can again use a
maximum likelihood estimate.  Thankfully, the work we did above did not go to
waste as the steps are the same up to Equation 6.  From there, we can substitute
the linear equation from Equation 9 in for :math:`\mu` and try to find the maximum
values for the vector of :math:`{\bf \beta}` values:

.. math::

    {\bf \beta}
    &= \arg\min_{\bf \beta} \sum_{i=1}^{n} (y_i-\beta_0 + \beta_1 x_{1} + ... + \beta_p x_{p})^2 \\
    &= \arg\min_{\bf \beta} \sum_{i=1}^{n} (y_i-\hat{y_i})^2
    \tag{11}

We use the notation :math:`\hat{y_i} = E(y|{\bf x}, {\bf \beta})` to denote the
predicted value (or expected value) of :math:`y` of our model.  Notice that the
estimate for the :math:`{\bf \beta}` values in Equation 11 is precisely the
equation for ordinary least squares estimates.  

I won't go into detail of how to solve Equation 11 but any of the standard
ideas will work such as a gradient descent or taking partial derivatives with
respect to all the parameters, set them to zero and solve the system of
equations.  There are a huge variety of ways to solve this equation that have
been studied quite extensively.

|h3| Prediction |h3e|

Once we have the coefficients for our linear regression from Equation 11, we
can now predict new values.  Given a vector of explanatory variables :math:`{\bf x}`,
predicting :math:`y` is a simple computation:

.. math::

    \hat{y_i} = E(y|{\bf x}) = \beta_0 + \beta_1 x_{1} + ... + \beta_p x_{p} \tag{12}

I included the expectation here to emphasize that we're generating a point
estimate for :math:`y`.  The expectation is the most likely value for :math:`y`
(according to our model) but our model is really predicting that :math:`y`
is most likely a band of values within a few :math:`\sigma` of this expectation.
To actually find this range, we would need to estimate :math:`\sigma` but it's
a bit outside the scope of this post.

Many times though, a point estimate is good enough and we can use it directly
as a new prediction point.  With classical statistics, you can also derive a
confidence interval or a prediction interval around this point estimate to gain
some insight into the uncertainty of it.  A full Bayesian approach is probably
better though since you'll explicitly state your assumptions (e.g. priors).


|h2| Generalized Linear Models (GLM) |h2e|

Changing up some of the modeling decision we made above, we get a different
type of regression model that is not any more complicated.
`Generalized linear models <https://en.wikipedia.org/wiki/Generalized_linear_model>`_
are a generalization of the ordinary linear regression model we just looked at above
except that it makes different choices.  Namely, the probability distribution and how
the mean of the outcome variable relates to the explanatory variables (i.e.
"link function").  The above methodology for deriving ordinary linear
regression can be equally applied to any of the generalized linear models.
We'll take a look at a `Poisson Regression
<https://en.wikipedia.org/wiki/Poisson_regression>`_ as an example.

|h3| Poisson Regression |h3e|

The first big difference between ordinary and Poisson regression is the distribution
of the outcome variable :math:`y`.  A Poisson regression uses a 
`Poisson distribution <https://en.wikipedia.org/wiki/Poisson_distribution>`_ (duh!) 
instead of a normal distribution:

.. math::

    Y \sim Poisson(\lambda) \\
    E(Y) = Var(Y) = \lambda \tag{13}

The Poisson distribution is a discrete probability distribution with a single
parameter :math:`\lambda`.  Since the Poisson regression is discrete,
so is our outcome variable.  Typically, a Poisson regression is used to
represent count data such as the number of letters of mail (or email) in a
day, or perhaps the number of customers walking into a store.

The second difference between ordinary and Poisson regressions is how we relate
the linear function of explanatory variables to the mean of the outcome
variable.  The Poisson regression assumes that the logarithm of the expected
value of the outcome is equal to the linear function of the explanatory
variables:

.. math::

    \log E(Y) = \log \lambda = \beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip} \tag{14}

Now with these two equations, we can again derive the log-likelihood function
in order to derive an expression to estimate the :math:`{\bf \beta}` parameters
(i.e. the maximum likelihood estimate).
Using the same scenario as Equation 6, namely :math:`n` :math:`(y_i, {\bf x_i})` 
i.i.d. points, we can derive a log likelihood function (refer to the Wikipedia
link for a reference of the probability mass function of a Poisson distribution):

.. math::

    \arg\max_{\bf \beta}  \mathcal{L(\beta|y_i)} 
        &= \prod_{i=1}^{n} \frac{\lambda^{y_i} e^{-\lambda}}{y_i!} \\
    \arg\max_{\bf \beta}  \log \mathcal{L(\beta|y)} 
        &= \sum_{i=1}^{n} \big( y_i \log\lambda - \lambda - \log{y_i!} \big) \\
        &= \sum_{i=1}^{n} \big( y_i \log\lambda - \lambda \big) \\
        &= \sum_{i=1}^{n} \big( y_i (\beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip}) 
            - e^{(\beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip})} \big) \tag{15}

You can arrive at the last line by substituting Equation 14 in.  Unlike ordinary
regression, Equation 15 doesn't have a closed form for its solution.  However, it is a convex
function meaning that we can use a numerical technique such as gradient descent
to find the unique optimal values of :math:`{\bf \beta}` that maximize the
likelihood function.

|h3| Prediction of Poisson Regression |h3e|

Once we have a point estimate for :math:`{\bf \beta}`, we can define the
distribution for our outcome variable:

.. math::

    Y \sim Poisson(exp\{\beta_0 + \beta_1 x_{1} + ... + \beta_p x_{p}\}) \tag{16}

and correspondingly our point prediction of :math:`\hat{y_i}` given its
explanatory variables:

.. math::

    \hat{y_i} = E(y_i) = exp\{\beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip}\} \tag{17}


|h3| Other GLM Models |h3e|

There are a variety of choices for the distribution of :math:`Y` and the
link functions.  This 
`table <https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function>`_
from Wikipedia has a really good overview from which you can derive the other
common types of GLMs.

The `logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ is
actually a type of GLM with outcome variable modeled as a 
`Bernoulli distribution <https://en.wikipedia.org/wiki/Bernoulli_distribution>`_
and link function as the `logit <https://en.wikipedia.org/wiki/Logit>`_
function (inverse of the `logistic function
<https://en.wikipedia.org/wiki/Logistic_function>`_, hence the name).
In the same way as we did for the ordinary and Poisson regression, you can
derive a maximum likelihood expression and numerically solve for the required
coefficients (there is no closed form solution similar to the Poisson regression).


|h2| Conclusion |h2e|

Linear regression is such a fundamental tool in statistics that sometimes
it is not explained in enough detail (or as clearly as it should be).
Starting from the bottom and building up a regression model is much more
interesting that the traditional method of presenting the end result and
scarcely relating it back to its probabilistic roots.  In my opinion, there's a
lot of beauty in statistics but only because it has its roots in probability.
I hope this post helped you see some of the beauty of this fundamental topic in
a new way.


|h2| References and Further Reading |h2e|

* Wikipedia: `Linear Regression <https://en.wikipedia.org/wiki/Linear_regression>`_, 
  `Ordinary Least Squares <https://en.wikipedia.org/wiki/Ordinary_least_squares>`_,
  `Generalized linear models <https://en.wikipedia.org/wiki/Generalized_linear_model>`_,
  `Poisson Regression <https://en.wikipedia.org/wiki/Poisson_regression>`_



.. [1] Since logarithm is monotonically increasing, it achieves the same maximum as the logarithm of a function at the same point.  It's also much more convenient to work with because many probability distributions have an exponents or are multiplicative.  The logarithm brings down the exponents and changes the multiplications to additions.

.. [2] This is true at least in a Bayesian interpretation.  In a frequentist interpretation, there is a fixed true value of :math:`\mu`, and what is random is the confidence interval we can find that "traps" it.  I've written a bit about it `here <link://slug/hypothesis-testing>`_.

.. [3] We explicitly use the conditional notation here because the value of :math:`y` depends on :math:`{\bf x}`.
