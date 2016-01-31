.. title: Normal Difference Distribution
.. slug: normal-difference-distribution
.. date: 2016-01-30 13:40:41 UTC-05:00
.. tags: normal, Gaussian, probability, marketing, mathjax
.. category: 
.. link: 
.. description: A walkthrough with theory and applications of the normal difference distribution.
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

   </h3>

.. |center| raw:: html

   <center>

.. |centere| raw:: html

   </center>

This post is going to look at a widely used probability distribution:
the normal difference distribution, or more explicitly the distribution that
results from the difference of two normally distributed random variables.
Any self respecting statistics class will teach students about the application
of this key distribution that has wide applicability across numerous fields such
as the economics, social sciences, and physical sciences to name a few.
As usual, we'll cover some of the theory about this distribution to gain some
intuition and apply it to a common problem in direct marketing of incremental
revenue.

.. TEASER_END

|h2| The Central Limit Theorem and The Normal Distribution |h2e|

Recall the `normal distribution
<https://en.wikipedia.org/wiki/Normal_distribution>`_ is a continuous
probability distribution parameterized by it's mean :math:`\mu`
and standard deviation :math:`\sigma`.  Its density function or PDF has the form:

.. math::

  f_N(x) = \frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \tag{1}

One of the reasons (among many) of why the normal distribution is so pervasive
is due to the `central limit theorem
<https://en.wikipedia.org/wiki/Central_limit_theorem>`_ (CLT), which roughly states
that the sample mean of a set of :math:`n` independent and identically
distributed (i.i.d.) random variables converge to a normal distribution as the
:math:`n` grows.  More precisely:

    Let :math:`{X_1, X_2, \ldots, X_n}` be :math:`n` i.i.d. random variables with
    :math:`E(X_i)=\mu` and :math:`Var(X_i)=\sigma^2` and  let 
    :math:`\bar{X} = \frac{X_1 + X_2 + \ldots + X_n}{n}` be the sample mean. 
    Then :math:`\bar{X}` approximates a normal distribution with mean of :math:`\mu` and
    variance of :math:`\frac{\sigma^2}{n}` for
    large :math:`n`, i.e. :math:`\bar{X} \approx N(\mu, \frac{\sigma^2}{n})`.

A key point here that cannot be stressed enough is that the i.i.d. random variable
:math:`X_i` can have *any* shape of distribution.  As long as we sample enough
of them, the average will converge to a normal distribution.  
`Wikipedia
<https://en.wikipedia.org/wiki/Illustration_of_the_central_limit_theorem>`_ has
a pretty good image showing this:

.. image:: /images/central_limit_theorem.png
   :height: 450px
   :alt: unit square
   :align: center

The top left shows the probability distribution of the original random variable 
:math:`X_i`.  This obviously doesn't look very "normal" with a finite `support
<https://en.wikipedia.org/wiki/Support_%28mathematics%29>`_ and sharp corners.
The next three images show the density functions of the summations of two,
three and four of the same variable.  You can see the fourth summation, we have
something that look qualitatively like a normal distribution even though what
we started out with was quite different.

This illustrates one of the key uses of a normal distribution: if we are trying
to estimate (i.e. "learn" or "guess") the mean (:math:`\mu`) of the underlying
random variable :math:`X_i` from a set of data points, we can use the sample
mean (:math:`\bar{X}`) as the estimator (i.e. our "best" guess) for the mean [1]_.

|h3| Estimators as Random Variables |h3e|

In my `last post <link://slug/hypothesis-testing>`_, I mentioned the fact that
estimators are random variables.  I just want to talk about this topic a bit
more because I think it's a pretty big point of confusion.
Imagine we have a population, for example a population of customers, that has
some sort of real-valued property associated with it like the spend of each
customer.  If we randomly draw a member of from this population, there is a
certain chance that this person will have a certain spend.  It's likely that
more customers will have spend values around $50 than $500.  We can model
this as a random variable :math:`X_i`.  A hypothetical distribution might look
something like this:

**TODO MAKE AN IMAGE OF GAMMA DISTRIBUTION for spend**

Usually we don't really have such a precise idea of distribution :math:`X_i`,
instead we need to infer it (or some property of it) from some data.  A common
thing to do is to estimate the mean of :math:`\mu = E(X_i)` given some sample
data points :math:`X_1, X_2, \ldots, X_n`.  As we mentioned above, an unbiased
estimator of this is the sample is :math:`\bar{X}`.  So far so good.

Here's a potentially confusing part: :math:`X_i` is a random variable that
represents a customer's spend from the population but after we sample and
observe it, its realization has a definite fixed non-random value.  For example,
customer :math:`i` might have a spend of $50 -- definitely not a random value.
Similarly, if we observe :math:`n` sample points and take their average to get
:math:`\bar{X}` we get a fixed, non-random value, for example, $56.87.
So how is it that :math:`X_i` and :math:`\bar{X}` are random variables?

One way to think about it is that even though customer :math:`i`'s spend
was $50 in this universe, imagine if we repeated the same experiment in a
parallel universe where everything was the same except for our random selection
for customer :math:`i`.  In that parallel universe, customer :math:`i` might
have a spend of $56.46.  Similarly, in that parallel universe, we might have
picked a different set of :math:`n` customers, so our sample mean :math:`\bar{X}`
might be slightly different, say $57.88.

If we look at :math:`X_i` and :math:`\bar{X}` this way, then we see that they are
"random".  So we want to derive some useful properties about 
them even before we know what their realization happens to be in this
universe.  Thus, we can analyze :math:`\bar{X}` as a random variable so that we can
derive some long-run behavior of it.  For example, we can compute a confidence
interval around :math:`\bar{X}` to guarantee that the "true value" of the mean,
:math:`\mu`, will be contained within it say 95% of the time.  I'll expand
more on this in the next section.  Before we go into that, let's take a look
at a realistic example of the sample mean.

.. admonition:: Example 1: Estimating Direct Marketing Average Customer Spend

   Let's suppose we're a brick and motar retail store trying to increase sales.
   One common way to do that is to send out some offers each week to get
   customers to come into our store to buy things.  This is usually called
   direct marketing.
   
   **First Question**: How do we estimate the average customer spend for
   someone who gets an offer?

   The intuitive -- and correct -- answer is to just tally up all the sales of
   people who got offers and divide by the number of people who go the offers.
   Simple enough but let's talk a bit about how we would do that and be a bit
   more pedantic about the details.

   Figuring out the number of people who go the offer is pretty simple -- you're
   sending out the offers!  So let's call this number :math:`n`.
   The harder part is figuring out which one of those people actually came to
   your store and spent money.  Some partial solutions to the problem are:

   * Attach a coupon to the offer.  Keep track of everyone who used the coupon
     as a proxy for people who spent money at your store.  This is imperfect
     solution because not everyone will actually use the coupon even though
     the offer could have caused them to come into the store (e.g.
     forgot/didn't see/doesn't care about the coupon).
   * Survey customers at checkout to see if they received the offer.  This is 
     pretty resource intensive (and annoying) so usually impractical.
   * Track all purchases with a loyalty card.  Since you the customers have a
     loyalty card, presumably you have their address and you know if you sent
     them an offer.  The downside is that not everyone will use a loyalty card
     even if they have one, and you can't measure people who don't have a
     loyalty card.  Customer Relationship Management (CRM) programs at
     retailers will usually will have dedicated offers to just loyalty members.
     This helps in ensuring that your measurement can be as accurate as
     possible but still imperfect if customers aren't forced to use their
     loyalty card at purchase.

   Let's assume that we're using the last option and this offer is from our CRM
   program.  So we can get a relatively accurate measure of average spend of
   all customers who go the flyer.

   **Next question**: how does this all relate to the CLT and the normal distribution? 
   (Here's the pedantic part; hopefully it will help with some intuition).

   First we assume that our population are all the people in our CRM program who
   could receive the offer, let's call it :math:`N`.  Theoretically, this
   population will have some distribution of spend.  For example, if I randomly
   sampled a person from this population, whose spend we'll call :math:`X_i`,
   there likely is a higher probability that he will spend $50 than $500.  The
   relative chance of each spend level will define the distribution of this
   population's spend.  This distribution is *unknown* distribution i.e. we do
   not know what :math:`X_i` looks like but we want to estimate the average
   spend :math:`\mu = E(X_i)`.

   Now going back to our offer, we sent this offer to :math:`n` random people
   in our CRM program, where most likely :math:`n < N` (because it would be too
   expensive to send an offer to everyone).  Presumably, each person's decision
   to shop at your store after receiving the offer is independent of everyone
   else.  In other words, the :math:`n` customers are independent and
   identically distributed according to :math:`X_i` (identically distributed because
   we randomly samples :math:`n` people from our distribution).

   Applying the CLT, the sample mean for these :math:`n` people will
   approximate be a normal distribution with :math:`\mu` and :math:`\sigma`
   according to unknown :math:`X_i`:

   .. math::

        \bar{X} = \frac{X_1 + X_2 + \ldots + X_n}{n}` \approx N(\mu, \frac{\sigma^2}{n}) \tag{2}
    
   Since :math:`\bar{X}` is a random variable, taking it's expected value:

   .. math::
    
        E(\bar{X}) \approx E(N(\mu, \frac{\sigma^2}{n})) = \mu \tag{3}
   
   Here we see that our sample mean is actually an unbiased estimate for our population
   mean.  Another way to think about it is, if repeatedly used this methodology
   to estimate the population mean :math:`\mu`, we would over-guess just as
   much as we would under-guess the true value of :math:`\mu`.  Theoretically
   this gives justification to our intuition that a simple average will give us
   a good estimate of the average customer spend of people who got the offer.

|h3| Confidence Interval for :math:`\bar{X}` |h3e|
   
As we saw the sample mean :math:`\bar{X}` is an unbiased point estimate for the
underlying population mean :math:`\mu`, but how good of an estimate is it?  For
that we need to compute a confidence interval.  In particular, we want to
compute an interval :math:`(a,b)` such that it "traps" the true value of
:math:`\mu` with some probability. 
Take a look at my `previous post <link://slug/hypothesis-testing>`_ for an
explanation of the intuition behind a confidence interval.

Let's start with a simplified unrealistic example: suppose we know the
underlying standard deviation of :math:`X_i`, represented by :math:`\sigma`.
Knowing this it is pretty simple to determine a confidence interval because as
we saw :math:`\bar{X} \approx N(\mu, \frac{\sigma}{\sqrt{n}})`.  Thus, :math:`\bar{X}`
will be normally distributed about :math:`\mu` with standard deviation
:math:`\frac{\sigma}{\sqrt{n}}`.  If we wanted to find, say an 95% confidence
interval, we could just use the pdf of our sample mean :math:`f_{\bar{X}}(x)`
and solve for a symmetric interval around the mean:

.. math::

    \int_{\mu - a}^{\mu + a} f_{\bar{X}}(x) dx = 0.95   \tag{4}

remembering that :math:`\bar{X}` is normally distributed.  Another common way to
approach this is to standardize (i.e. shift and scale) the normal variable to
the standard normal variable :math:`Z \sim N(0, 1)` with zero mean and unit
variance.  This is pretty simple by using just a change of variables.
Let's go through the steps just to make sure we understand what's going on.
Define:

.. math::

    Z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \tag{5}

And with some manipulation (remember :math:`\mu` and :math:`\sigma` are some
fixed real number parameterizing our distribution :math:`X_i`):

.. math::

    P(\bar{X} \leq s) &= P(\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \leq
                      \frac{s - \mu}{\frac{\sigma}{\sqrt{n}}}) \\
                  &= P(Z \leq \frac{s - \mu}{\frac{\sigma}{\sqrt{n}}}) \\
    &= \int_{-\infty}^{\frac{s - \mu}{\frac{\sigma}{\sqrt{n}}}} 
            \frac{1}{\frac{\sigma^2}{n}\sqrt{2\pi}} 
            exp\{-\frac{(x-\mu)^2}{2\frac{\sigma^2}{n}}\}dx \tag{6}

Change variables in the integral with :math:`x = \frac{\sigma}{\sqrt{n}}z +
\mu`, we get the standard normal distribution for :math:`Z`:

.. math::

    P(Z \leq z) =  \int_{-\infty}^{z} \frac{1}{\sqrt{2\pi}} exp\{-\frac{z^2}{2}\}dx \tag{7}

Starting with a 95% confidence interval for a standard normal distribution and working
back to our sample mean:

.. math::

    P(-z \leq Z \leq z) &= 0.95 \\
    &= P(-1.96 \leq \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \leq 1.96) \\
    &= P(\bar{X} - 1.96\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + 1.96\frac{\sigma}{\sqrt{n}}) 
    \tag{8}

We get a 95% confidence interval for :math:`\mu`: :math:`(\bar{X} - 1.96\frac{\sigma}{\sqrt{n}}, \bar{X} + 1.96\frac{\sigma}{\sqrt{n}})`.

However, remember we used a big simplifying assumption: we know the value of :math:`\sigma`.
In most realistic cases, this is an unknown quantity that we also have to estimate.
It turns out an unbiased estimator for the variance :math:`\sigma^2` is the
`sample variance <https://en.wikipedia.org/wiki/Variance#Sample_variance>`_:

.. math::

    s^2 = \frac{1}{n-1} \Sigma_{i=1}^{n} (x_i - \bar{x})^2 \tag{9}




|h2| Difference of Two Normal Distributions |h2e|




|h2| Incremental Sales in Direct Marketing |h2e|

|h2| Conclusion |h2e|

|h2| References and Further Reading |h2e|

* Wikipedia: `Sum of normally distributed random variables <https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables>`_, `Central Limit Theorem <https://en.wikipedia.org/wiki/Central_limit_theorem>`_, `Sample Variance <https://en.wikipedia.org/wiki/Variance#Sample_variance>`_
* 


|br|
|br|

.. [1] In fact the sample mean :math:`\bar{X}` is an `unbiased estimator <https://en.wikipedia.org/wiki/Bias_of_an_estimator>` for the mean :math:`\mu` of :math:`X_i`.  One way of thinking about bias is how often your estimate will be above or below the true value of what you're trying to estimate.  If in the long run (say after thousands of difference instances where you tried to estimate :math:`\mu`), your estimate is above the true value of :math:`\mu` just as often as it is below the true value, then you have an unbiased estimator.  Similarly, if you are systematically guessing "too high" or "too low" then you probably have a biased estimator.


