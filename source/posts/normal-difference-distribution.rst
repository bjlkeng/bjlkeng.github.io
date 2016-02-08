.. title: Elementary Direct Marketing Statistics
.. slug: normal-difference-distribution
.. date: 2016-01-30 13:40:41 UTC-05:00
.. tags: normal, Gaussian, probability, direct marketing, mathjax
.. category: 
.. link: 
.. description: A primer on some elementary statistics for direct marketing.
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

This post is going to look at some elementary statistics for direct marketing.
Most of the techniques are direct applications of topics learned in a first
year statistics course (so nothing too complicated yet :p).  I'll start off by
covering some background and terminology on the direct marketing and then
introduce some of the statistical inference techniques that are commonly used.
As usual, I'll try to mix in some theory where appropriate to build some intuition.

.. TEASER_END

|h2| Direct Marketing |h2e|

`Direct marketing <https://en.wikipedia.org/wiki/Direct_marketing>`_ is a form
of advertising that sends communications directly to potential customers through
a wide variety of media such as (snail) mail, e-mail, text message, websites
among others.  The distinguishing feature of direct marketing efforts is that a
business has a database of potential customers where the business will send a
direct (usually personally addressed) message to these customers (or a segment
of them) with a specific outcome in mind.  Some familiar examples include: 

* Signing up for a rewards plan at a retailer and end up emails about offers
  periodically at the store (e.g. "Buy $75 get $25 off")
* A (snail) mail offer for a "discounted" magazine subscription (usually they
  get your address from another magazine you've subscribed to).
* A telemarketer calls you (at what seems to always be an inconvenient time)
  to get you to buy their product.

The interesting thing about direct marketing campaigns (as opposed to mass
`market campaigns <https://en.wikipedia.org/wiki/Mass_marketing>`_) is that 
you can individually track the behavior of each message that you send out.
This fine grained tracking allows you to scientifically (read:
statistically) measure the effectiveness of a certain direct marketing campaign
and allows you to learn what works and what doesn't.

In the following sections, I'll direct marketing campaigns with respect
to retail direct marketing campaigns because that's what I'm most familiar
with, but the general idea should apply to other domains.

|h3| Direct Marketing Campaigns |h3e|

Imagine you are a retailer wanting to increase sales.  With very broad strokes,
there are three ways to go about it:

1. Get more (unique) customers to come in the door (or onto your website).
2. Get each customer to spend more money on each visit.
3. Get each customer to come back more often.

Obviously there are many ways to do this and it is a very complex subject, so
let's stick with one of the most popular ways: by providing an offer to the
customer.  The offer might entice a new customer to walk in, stretch
an existing customer to spend a bit more, or even cause a previous customer to
come back earlier than she would have without the offer.

The simplest type of offer is a mass-promotion where the offer is available to
everyone for example "$2 off a bag of milk".  The drawback of giving it to
everyone is that some people would have bought the milk without the mass-offer
(resulting in a missed opportunity for revenue), while for others $2 may not
be enough of a bargain.  Enter direct marketing.

Now suppose you have a database of all (or a large segment) of your customers.
You might have their email address, phone number, age, purchase history,
address, and any number of other individual attributes.  Now you can target
customers in a more fine grained way.  You might only want to give that $2 milk deal
to the price-sensitive shoppers so that they don't go over to your competitor while
also trying to `cross-sell <https://en.wikipedia.org/wiki/Cross-selling>`_ them
other complementary products.  Or you might want to target your high-end
customers by `upselling <https://en.wikipedia.org/wiki/Upselling>`_ them with
an offer for a luxury version of products they previously bought.
In the limit, you can customize the offer on a one-to-one basis for true 
`personalization <https://en.wikipedia.org/wiki/Personalization>`_.

There are many machine learning and statistical techniques to build models to
target and predict how customers will respond to the various combinations of
offers.  I'll cover that in another post, for now let's stick with
something more mundane (but perhaps more important): after sending out a direct 
marketing campaign, how do I know if it worked?

|h3| Important Metrics |h3e|

The most important aspect when running a direct marketing campaign is make sure
it is achieving your business objective, which usually relates to one of the
following: incremental revenue, incremental active rate (similar to a
conversion rate), or return on investment (ROI).  We'll focus on the former, talk a
bit about the second one, and leave the last one out for now.

Let's try to translate our business objective into something that we can
measure.  Increasing incremental revenue in a direct marketing campaign means
that by running this marketing campaign, we will have made more money than not
running this campaign.  Incremental active rate means that we will have
convinced more (unique) customers to walk in the door than we would have
without providing this offer to them.  Lastly, ROI is simply the efficiency
of the campaign dollars, if you have negative ROI, you're basically losing
money running this campaign.

Revenue, active rate and ROI can be measured across campaigns using some simple ratios:

.. math::

    \text{spend per customer (SPC)} &= \frac{\text{total revenue}}{\text{customers who received offer}} \\
    \text{active rate} &= \frac{\text{unique customers}}{\text{customers who received offer}} \\
    \text{ROI} &= \frac{\text{revenue} - \text{cost of campaign}}{\text{cost of campaign}} \tag{1}

The difficulty question is how can you measure the effectiveness of the offer
because you can't simultaneously give the offer to a person *and* not give it
to them at the same time.  This is actually a well solved problem used in the
medical field to determine the effectiveness of treatments for decades.

|h3| Control, Control, Control |h3e|

The experimental setup involves dividing your population (e.g. customers whom
you send the offer to) into two randomly selected groups: treatment and
control.  The treatment group receives the direct marketing offer (or drug in
the case of medical trials) and the control group will receive the
business-as-usual placebo.  Depending on what you are trying to measure the
business-as-usual treatment might be nothing (measuring the overall
effectiveness of the offer), a generic offer (measuring the A vs. B
effectiveness of your offer), or some variation that allows you to compare one
treatment to another.

The randomized control group allows us the "control" for confounding variables.
That is, for hidden biases that might be introduced when running the experiment [1]_
such as a holiday or perhaps a competitor's sale.
It also allows us to make causal statements about the relationship between two
variables.  Instead of just saying treatment A is correlated or associated with
a sales increase, we can say treatment A *causes* a sales increase.
Randomized control groups are the primary method in which we can verify causality
between two events.

A few important points when practically designing an experiment in this
scenario [2]_:

* A randomly selected control group is taken in order to make a causal statement.
* The samples (i.e. customers) are independent and measured on an individual
  basis (not just the total revenue but the revenue for each customer too).
* Sample size is large enough (for both groups) to have a statistically
  significant conclusion [3]_.

We'll cover more on these topics below.

|h2| The Math |h2e|

Now that we have a high level understanding of how direct marketing campaigns
work, let's try to work out some of the math.  

Let's imagine we have we have :math:`n` `i.i.d.
<https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`_
variables :math:`Y_1, Y_2, \ldots, Y_n` representing our outcome variables
i.e. :math:`Y_i` is customer :math:`i`'s total spend at your store during the
campaign period.  
Also denote the binary treatment variable as :math:`X_i=1` as "treated" (given
the offer) and :math:`X_i=0` as "not treated" (control group or not given the
offer) for the :math:`i^{th}` customer.

To not bury the lead, an good estimator for measuring campaign effectivness is
just using the difference of spend per customer or **lift**  of the
treatment and control group (i.e. different of population means):

.. math::

    \hat{\text{lift}} &= \hat{E}[Y|X=1] - \hat{E}[Y|X=0] \\
                      &= \bar{Y}_{X=1} - \bar{Y}_{X=0} \\
                      &= \frac{1}{n_{X=1}} \Sigma_{i=1}^{n} Y_i X_i
                         - \frac{1}{n_{X=0}} \Sigma_{i=1}^{n} Y_i (1 - X_i) \\
                      &= SPC_{\text{treatment}} - SPC_{\text{control}}          \tag{2}

where the :math:`\hat{}` symbol represents an estimate,
:math:`n_{X_0} = \Sigma_{i=1}^{n} X_i` and :math:`n_{X_1} = \Sigma_{i=1}^{n} (1-X_i)`.
All those equations boil down to basically just taking the difference of spend
per customer between treatment and control.  In some of the more math heavy
sections below, you'll see why we introduced all this notation.
First, let's see why our control groups have to be a random.

|h2| Causal Inference [4]_ |h2e|

As I mentioned before, we can never simultaneously give a promotional offer
*and* not give it at the same time (unless we had some kind of parallel universe).  
However, the ideal measurment for campaign effectiveness is exactly this
quantity!  Let's define this more precisely.

Introduce two variables :math:`C_{X=0}, C_{X=1}` as potential outcome
variables.  `C_{X=0}` is the outcome if we did not treat customer :math:`i` and
`C_{X=1}` is the outcome if we did treat customer :math:`i`.
Therefore:

.. math::

 Y = 
 \begin{cases} 
      C_{X=0} & X = 0 \\
      C_{X=1} & X = 1
 \end{cases}  \tag{3}

Or more concisely, :math:`Y=C_X`.  
(Note: we can never observe both :math:`C_{X=0}` and :math:`C_{X=1}` at the same time.)

The actual effect we want to measure is actually the difference in expected
value of these two variables called the **average causal effect**:

.. math::

    lift = E(C_{X=1}) - E(C_{X=0}) \tag{4}

In other words, we want to find the SPC of sending *everyone* an offer minus
the SPC of *not* sending everyone an offer.
Equation 2 looks similar but actually measures something different called
**association** (denoted by :math:`\alpha`):

.. math::

    \alpha = E[Y|X=1] - E[Y|X=0]  \tag{5}

It's a widely known fact that association does not equal causation.
Let's take a look at a small example why.

.. admonition:: Example 1


 .. math::

    \newcommand\T{\Rule{0pt}{1em}{.3em}}
    \begin{array}{|c|c|c|c|}
    \hline X & Y & C_{X=0} & C_{X=1} \T \\\hline
      0  & $0 & $0 & $0 \\\hline
      0  & $0 & $0 & $0 \\\hline
      0  & $0 & $0 & $0 \\\hline
      0  & $0 & $0 & $0 \\\hline
      1  & $10 & $10 & $10 \\\hline
      1  & $10 & $10 & $10 \\\hline
      1  & $10 & $10 & $10 \\\hline
      1  & $10 & $10 & $10 \\\hline
    \end{array}

 Here we can can calculate the lift and association:

 .. math::

    \alpha &= \frac{$10 + $10 + $10 + $10}{4} - \frac{$0 + $0 + $0 + $0}{4} \\
           &= \frac{$10} \\
           \\
    \text{lift} &= \frac{$0 + $0 + $0 + $0 + $10 + $10 + $10 + $10}{8} \\
     & - \frac{$0 + $0 + $0 + $0 + $10 + $10 + $10 + $10}{8} \\
                &= $0 \tag{6}

 The lift is zero because the treatment has no effect: look at the hypothetical
 :math:`C_X` variables, they are the same regardless of whether or not the
 treatment was applied.  The association on the other hand is clearly positive
 at $10.

Coming up with other examples where :math:`\alpha > 0` but :math:`\text{lift} < 0` and
related combinations are not to difficult.  The reason why we got such different
values for :math:`\alpha` and :math:`\text{lift}` is because :math:`C_{X=0}, C_{X=1}`
are not independent of :math:`X`.  That is the treatment is not independent of the 
customer, in the example, we put all the high value customers in the treatment group
while putting the low value ones in the "control" group.

.. admonition:: **Theorem 1**

 If we randomly assign subjects to treatment and control such that :math:`P(X=0) > 0`
 and :math:`P(X=1) > 1`, then :math:`\alpha=\text{lift}`.
    
 **Proof**
     Since X is randomly assigned, X is independent of :math:`C_{X=1}, C_{X=0}`, so:
 
     .. math::
     
         \text{lift} &= E(C_{X=1}) - E(C_{X=0}) \\
                     &= E(C_{X=1}|X=1) - E(C_{X=0}|X=0) && \text{since } X \text{ is independent of } C_X \\
                     &= E(Y|X=1) - E(Y|X=0) && \text{since } Y = C_X \\
                     &= \alpha   \tag{7}







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

* Wikipedia: `Direct marketing <https://en.wikipedia.org/wiki/Direct_marketing>`_
* `All of Statistics: A Concise Course in Statistical Inference <http://link.springer.com/book/10.1007%2F978-0-387-21736-9>`_ by Larry Wasserman.
* `Data Mining Techniques: For Marketing, Sales, and Customer Relationship Management <http://www.amazon.com/Data-Mining-Techniques-Relationship-Management/dp/0470650931/>`_ by Linoff
  
  
  `Sum of normally distributed random variables <https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables>`_, `Central Limit Theorem <https://en.wikipedia.org/wiki/Central_limit_theorem>`_, `Sample Variance <https://en.wikipedia.org/wiki/Variance#Sample_variance>`_



|br|
|br|


.. [1] An interesting story I read about in `The Emperor of All Maladies <https://en.wikipedia.org/wiki/The_Emperor_of_All_Maladies>`_ about how several large scale trials testing the effectiveness of mammograms (low-dose x-ray imaging to detect breast cancer) were undone by implicit selection biases.  In one of the Canadian studies, nurses would write down trial patients names in a notebook where the first line corresponded to the treatment group, second control group, third treatment and so forth.  The nurses administering the trial subtlely biased the results by feeding patients who they thought were more in need to the treatment group.  A compassionate gesture but, statistically, a failed experiment.  Without the benefit of a truly randomized trial, they could not longer analyze the effectiveness of the treatment in insolation of confounding variables.

.. [2] The topic of design of `randomized control trials <https://en.wikipedia.org/wiki/Randomized_controlled_trial>`_ can is actually be quite complex.  In this explanation, we're assuming relatively large population sizes (10s of thousands), which allows us to make a lot of simplifying assumptions.  When dealing with small sample sizes, the randomness of the samples can be quite large making it much more important to design your experiment properly so you can get proper conclusions.  For larger population sizes, we usually worry less about this.

.. [3] Many marketers are tempted to *not* take a control group.  Their reasoning is something along the lines of "but I'm missing out on sales!", in which you should respond back "how do you know that?"  It's quite possible the offer could have absolutely no meaningful effect on your customers (e.g. targeting wrong product to people who don't want it) and possibly a negative effect (e.g. the discount may be too high with not enough people coming in)!  Just because you're giving people a discount doesn't mean it always increases sales.  Further, even if you have an increase, you don't know *how much* of an increase it is by i.e. the effect size.  If you different offers boosted sales by 1% vs. 10%, you should know that!  This is how you can test and learn to improve your overall business.

.. [4] This section was primarily based on *All of Statistics*, Chapter 16.  It has a great explanation of how randomized control groups work.  Check it out if my quick explanation glosses over too many things.
