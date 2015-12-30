.. title: An Introduction to Statistical Inference and Hypothesis Testing
.. slug: hypothesis-testing
.. date: 2015-12-29 10:22:26 UTC-05:00
.. tags: hypothesis testing, models, mathjax
.. category: 
.. link: 
.. description: A post explaining hypothesis testing in a (hopefully) easy to understand way.
.. type: text

.. |br| raw:: html

   <br />

.. |H2| raw:: html

   <h3>

.. |H2e| raw:: html

   </h3>

.. |H3| raw:: html

   <h4>

.. |H3e| raw:: html

   </hk>

Introduction

.. TEASER_END

|h2| Statistical Models and Inference |h2e|

Before we begin talking about (statistical) hypotheses, it's important to clear
up some common statistics terms and ideas that are sometimes casually thrown
around.  I, for one, did not have a rigorous understanding of these concepts
and definitely not a very good intuitive sense.  Let's try to explain it with
enough math to get a good intuition about the subject by starting with some of
the "big" ideas then onto some more precise definitions.

|h3| A Couple of Big Ideas |h3e|

The first big idea is that *all data (or observations as statisticians
like to say) have a "true" probability distribution* [1]_ .  Of course, it may
never be possible to precisely define this distribution because the real world
doesn't fit so nicely into the distributions we regularly learn in stats class.

The second big idea is that **statistical inference** [2]_ (or as computer scientists
call it "learning" [3]_) basically boils down to estimating this distribution
directly by computing the distribution or density function [4]_, or indirectly by
estimating derived metrics such as the mean or median of the distribution.  A
typical question we might ask is:

    Give a sample :math:`X_1, X_2, \ldots, X_n` drawn from a distribution :math:`F`,
    how do we estimate :math:`F` (or some properties of :math:`F`)?

Of course there are variations to this question depending on the precise
problem such as regression but by and large view it comes down to finding
things about :math:`F` (or some derived properties).

|h3| Models, models, models |h3e|

Now that we have those two big ideas out of the way, let's define a
(statistical) model:

    A **statistical model** :math:`\mathfrak{F}` is a set of distributions (or
    densities or regression functions).

The idea here is that we want to define a subset of all possible distributions
that closely approximates the "true" distribution (whether or not
:math:`\mathfrak{F}` actually contains :math:`F` [5]_).  By far, the most
common type of model is a **parametric model**, which defines :math:`\mathfrak{F}`
using a finite number of parameters.  For example, if we assume that the
data comes from a Normal distribution, we would use the parametric model as such:

.. math::

  \mathfrak{F} = \big\{ f(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \mu \in \mathbb{R}, \sigma > 0 \big\}

Here we use the notation :math:`f(x; \mu, \sigma)` to denote a density function
of :math:`x` parameterized by :math:`\mu` and :math:`\sigma`.  Similarly, when
we have data of the form :math:`(X_i, Y_i)` and we want to learn regression
function :math:`r(x) = E(Y|X)`, we could define a model for
:math:`\mathfrak{F}` to be all functions of :math:`x`, :math:`r(x)`, that are
straight lines.  This gives us a linear regression model.

The other type of model is a **non-parametric model**.  Here the number of
parameters is not finite or fixed by the model, instead the model is defined by
the input data.  In essence, the parameters are determined by the training data
(not the model).  For example, a histogram can be thought of as a simple
non-parametric model that estimates a probability distribution because the data
determines the shape of the histogram.  

Another example would be a k-nearest neighbor algorithm that can classify a new
observation solely based on its k-nearest neighbors from training data.  The
surface defined by the classification function is not pre-defined rather it is
determined soley by the training data (and hyper parameter :math:`k`).  You can
contrast this with a logistic regression as a classifier, which has a rigid
structure regardless of how well the data matches. 

Although, it sounds appealing to let the "data define the model", non-parameteric
data typically requires a much larger sample size to draw a similar conclusion
compared to parametric methods.  This makes sense intuitively since parametric
methods have the advantage of having the extra model assumptions, so making
conclusions should be easier all else being equal.  Of course, you must be
careful picking the *right* parametric model or else your conclusions from the
parametric model might be invalid.

|h3| Types of Statistical Inference |h3e|




|h2| References and Further Reading |h2e|

* `All of Statistics: A Concise Course in Statistical Inference <http://link.springer.com/book/10.1007%2F978-0-387-21736-9>`_ by Larry Wasserman. (available free online)
* Wikipedia: `Statistical models <https://en.wikipedia.org/wiki/Statistical_model>`_, `Statistical Inference <https://en.wikipedia.org/wiki/Statistical_inference>`_, `Nonparametric Statistics <https://en.wikipedia.org/wiki/Nonparametric_statistics>`_, TODO.



.. [1] Taking note that no model can truly represent the reality leading to the aphorism: `All models are wrong <https://en.wikipedia.org/wiki/All_models_are_wrong>`_.

.. [2] `Inferential statistics <https://en.wikipedia.org/wiki/Statistical_inference>`_ is in contrast to `descriptive statistics <https://en.wikipedia.org/wiki/Descriptive_statistics>`_, which only tries to describe the sample or observations -- not estimate a probability distribution.  So examples are measures of central tendency like mean or median, or variability such as standard deviation or min/max values.  Note that although the mean of a sample is a descriptive statistic, it is also an estimate for the expected value of a given distribution, thus used in statistical inference.  Similarly for the other descriptive statistics.

.. [3] There is a great chart in *All of Statistics* that shows the difference between statistics and computer science/data mining terminology on page xi of the preface.  It's very illuminating to contrast the two especially since terms like estimation, learning, covariates, hypothesis are thrown around very casually in their respective literature.  I come more from a computer science/data mining and learned most of my stats afterwards so it's great to see all these terms with their definitions in one place.

.. [4] Might be obvious but let's state it explicitly: *distribution* refers to the cumulative distribution function (CDF), and *density* refers to the probability density function (PDF).

.. [5] In fact, most of the time :math:`\mathfrak{F}` will not contain :math:`F` since as we mentioned above, the "true" distribution is probably much more complex than any model we could come up with.
