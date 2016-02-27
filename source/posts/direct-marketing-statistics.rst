.. title: Elementary Statistics for Direct Marketing
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

   </h4>

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
Note: we'll represent random variables with capital letters  and their corresponding values after they have been observed with
lower case letters [4]_.

To not bury the lead, a good point estimator for measuring campaign
effectivness is just using the difference of spend per customer (SPC) or
**lift**  of the treatment and control group (i.e. different of population
means):

.. math::

    \hat{\text{lift}} &= \hat{E}[Y|X=1] - \hat{E}[Y|X=0] \\
                      &= \bar{y}_{X=1} - \bar{y}_{X=0} \\
                      &= \frac{1}{n_T} \Sigma_{i=1}^{n} y_i x_i
                         - \frac{1}{n_C} \Sigma_{i=1}^{n} y_i (1 - x_i) \\
                      &= SPC_{\text{treatment}} - SPC_{\text{control}}          \tag{2}

where the :math:`\hat{}` symbol represents an estimate,
:math:`n_T = \Sigma_{i=1}^{n} x_i` and :math:`n_C = \Sigma_{i=1}^{n} (1-x_i)`.
All those equations boil down to basically just taking the difference of spend
per customer between treatment and control.  In some of the more math heavy
sections below, you'll see why we introduced all this notation.
First, let's see why our control groups have to be a random sample.

|h3| Causal Inference [5]_ |h3e|

As I mentioned before, we can never simultaneously give a promotional offer
*and* not give it at the same time (unless we had some kind of parallel universe).  
However, the ideal measurment for campaign effectiveness is exactly this
quantity!  Let's define this more precisely.

Introduce two variables :math:`C_{X=0}, C_{X=1}` as potential outcome
variables.  :math:`C_{X=0}` is the outcome if we did not treat customer :math:`i` and
:math:`C_{X=1}` is the outcome if we did treat customer :math:`i`.
Therefore:

.. math::

 Y = 
 \begin{cases} 
      C_{X=0} & \text{when }X = 0 \\
      C_{X=1} & \text{when }X = 1
 \end{cases}  \tag{3}

Or more concisely, :math:`Y=C_X`.  
(Note: we can never observe both :math:`C_{X=0}` and :math:`C_{X=1}` at the same time.)

The actual effect we want to measure is actually the difference in expected
value of these two variables called the **average causal effect**:

.. math::

    \text{lift} = E(C_{X=1}) - E(C_{X=0}) \tag{4}

In other words, we want to find the SPC of sending *everyone* an offer minus
the SPC of *not* sending everyone an offer.
Equation 2 looks similar but actually measures something different called
**association** (denoted by :math:`A`):

.. math::

    A = E[Y|X=1] - E[Y|X=0]  \tag{5}

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

    A &= \frac{$10 + $10 + $10 + $10}{4} - \frac{$0 + $0 + $0 + $0}{4} \\
           &= $10 \\
           \\
    \text{lift} &= \frac{$0 + $0 + $0 + $0 + $10 + $10 + $10 + $10}{8} \\
     & - \frac{$0 + $0 + $0 + $0 + $10 + $10 + $10 + $10}{8} \\
                &= $0 \tag{6}

 The lift is zero because the treatment has no effect: look at the hypothetical
 :math:`C_X` variables, they are the same regardless of whether or not the
 treatment was applied.  The association on the other hand is clearly positive
 at $10.

Coming up with other examples where :math:`A > 0` but :math:`\text{lift} < 0` and
other related combinations is not too difficult.  The reason why we got such different
values for :math:`A` and :math:`\text{lift}` is because :math:`C_{X=0}, C_{X=1}`
are not independent of :math:`X`.  That is, the treatment is not independent of the 
customer.  In the above example, we put all the high value customers in the treatment group
while putting the low value ones in the control group.

.. admonition:: **Theorem** 

 If we randomly assign subjects to treatment and control such that :math:`P(X=0) > 0`
 and :math:`P(X=1) > 1`, then :math:`A=\text{lift}`.
    
 **Proof**
     Since X is randomly assigned, X is independent of :math:`C_{X=1}, C_{X=0}`, so:
 
     .. math::
     
         \text{lift} &= E(C_{X=1}) - E(C_{X=0}) \\
                     &= E(C_{X=1}|X=1) - E(C_{X=0}|X=0) && \text{since } X \text{ is independent of } C_X \\
                     &= E(Y|X=1) - E(Y|X=0) && \text{since } Y = C_X \\
                     &= A   \tag{7}


Using Theorem 1 we can see that by assigning random control groups, our
use of difference in SPC (i.e. association) using Equation 2 is identical
the actual causal effect i.e. lift.  However, if we
don't have random assignments (and have some kind of bias in assignment towards
treatment or control) then there is no guarantee that the association we
computed in Equation 2 has anything to do with lift, as we saw in Example 1.


|h3| Central Limit Theorem and Confidence Intervals |h3e|

To simplify our notation, let's define :math:`U_i = Y_i X_i` 
and :math:`V_i = Y_i (1 - X_i)` to represent samples from the treatment
and control respectively.  Let their respective mean and variance be represented by
:math:`\mu_U`, :math:`\mu_V` and :math:`\sigma^2_U`, :math:`\sigma^2_V`.  This is
done just for convenience so we don't have to keep writing our equations with
:math:`X_i` in them.

In general, the distributions of customer spend, :math:`U_i` and :math:`V_i`,
do not take any familiar form of distribution that we know.  However, using the
results of the 
`central limit theorem (CLT) <https://en.wikipedia.org/wiki/Central_limit_theorem>`_, 
we know that the sample mean of each population (treatment :math:`\bar{U}` and
control :math:`\bar{V}`) can be approximated by a normal distribution:
 
.. math::

    \bar{U} \approx N(\mu_U, \frac{\sigma^2_U}{n_U}) \\
    \bar{V} \approx N(\mu_V, \frac{\sigma^2_V}{n_V}) \tag{8}

Usually our :math:`n` is quite large (:math:`n>10,000`) so our normal
approximations are quite good.
Similarly, using the strong `law of large numbers <https://en.wikipedia.org/wiki/Law_of_large_numbers>`_,
the `sample variance <https://en.wikipedia.org/wiki/Variance#Sample_variance>`_
(denoted by :math:`s^2`) is a pretty good approximation of the actual variance
when we have large :math:`n`:

.. math::

    \sigma_U^2 &\approx s_U^2 \\
    \sigma_V^2 &\approx s_V^2 \tag{9}

Using Equation 8 and 9, we get our approximation of the sample mean for large :math:`n`:
    
.. math::

    \bar{U} &\approx N(\bar{u}, \frac{{s}^2_U}{n_U}) \\
    \bar{V} &\approx N(\bar{v}, \frac{{s}^2_V}{n_V})  \tag{10}

Knowing that the `difference of two normal distributions
<http://mathworld.wolfram.com/NormalDifferenceDistribution.html>`_ is just a
normal distribution (with mean equal to the difference of the means, and
variance equal to the sum of variances), our lift is:


.. math::

     \text{lift} &= \bar{U} - \bar{V} \\
                 &= N(\mu_{U}, \frac{\sigma^2_U}{n_U}) - N(\mu_{V}, \frac{\sigma^2_V}{n_V}) \\
                 &= N(\mu_{U} - \mu_{V}, \frac{\sigma^2_U}{n_U} + \frac{\sigma^2_V}{n_V}) \\
                 &\approx N(\bar{u} - \bar{v}, \frac{s^2_U}{n_U} + \frac{s^2_V}{n_V}) \tag{11} 

Now that our lift is simply just a normal random variable whose mean and variance we know how to estimate,
we can get a :math:`1 - \alpha` two sided confidence interval.  
Since lift is approximately normal,
we know that :math:`Z=\frac{\text{lift} - \mu_{\text{lift}}}{\sigma_{\text{lift}}}` 
has a standard normal distribution N(0, 1):

.. math::

    P(-z_{\alpha/2} &\leq Z \leq Z_{\alpha/2}) = 1 - \alpha \\
    P(-z_{\alpha/2} &\leq \frac{{\text{lift}} - \mu_{\text{lift}}}{{\sigma}_{\text{lift}}} \leq z_{\alpha/2}) = 1 - \alpha \\
    P(-z_{\alpha/2} {\sigma}_{\text{lift}} &\leq {\text{lift}} - \mu_{\text{lift}} \leq z_{\alpha/2} {\sigma}_{\text{lift}}) = 1 - \alpha \\
    P(-z_{\alpha/2}{{\sigma}_{\text{lift}}} - {\text{lift}} &\leq - \mu_{\text{lift}} \leq z_{\alpha/2}{{\sigma}_{\text{lift}}} - {\text{lift}}) = 1 - \alpha \\
    P({\text{lift}} - z_{\alpha/2}{{\sigma}_{\text{lift}}} &\leq \mu_{\text{lift}} \leq {\text{lift}} + z_{\alpha/2}{{\sigma}_{\text{lift}}}) = 1 - \alpha \tag{12}

That is, the true lift (:math:`\mu_{\text{lift}}`) lies in the interval
:math:`[{\text{lift}} - z_{\alpha/2}{{\sigma}_{\text{lift}}}, {\text{lift}} - z_{\alpha/2}{{\sigma}_{\text{lift}}}]`,
:math:`1-\alpha` of the time.   Plugging in our estimates of :math:`\hat{\text{lift}}=\bar{u}-\bar{v} = SPC_{\text{T}} - SPC_\text{C}` 
and :math:`\hat{\sigma_{\text{lift}}}=\sqrt{\frac{s^2_U}{n_U} + \frac{s^2_V}{n_V}}` and looking up the
appropriate `Z-score <https://en.wikipedia.org/wiki/Standard_score>`_, we can
compute our :math:`1-\alpha` confidence interval:

.. math::

    [SPC_{\text{T}} - SPC_\text{C} - z_{\alpha/2}\sqrt{\frac{s^2_U}{n_U} + \frac{s^2_V}{n_V}},
     SPC_{\text{T}} - SPC_\text{C} + z_{\alpha/2}\sqrt{\frac{s^2_U}{n_U} + \frac{s^2_V}{n_V}}]  \tag{13}




|h3| Activation Rate and Binomial Outcome Variables |h3e|

All the math above equally applies to when your outcome variable is not a
real-valued number but a binary outcome such as activation or conversion.
In that case, each customer can only have two outcomes: :math:`1` (shops or
converts) and :math:`0` (doesn't shop or convert).  There are a few caveats
though.

A good rule of thumb of when you can use a normal to approximate a binomial
is (from Wackerly et.  al):

.. math::

    n \geq 9 \frac{\text{larger of }p\text{ and }(1-p)}{\text{smaller of }p\text{ and }(1-p)} \tag{14}

So for :math:`p=0.01`, :math:`n \geq 9 \frac{0.99}{0.01} = 891`, meaning you
want your treatment and control groups to be at least 900.  Most likely you
will want bigger values of :math:`n` to control the error rate (see below).


The standard unbiased estimators for mean and variance of a binomial would be
used where :math:`Y` is the number of successes (or :math:`1`'s) in the
sample, and :math:`n` is the total number of samples:

.. math::

    \hat{\mu_\bar{Y}} &= \hat{p_Y} = \frac{y}{n} \\
    \hat{\sigma^2} &= \frac{\hat{p}(1-\hat{p})}{n} \tag{15}

Using our lift notation above, we would get:

.. math::

    \hat{\text{lift}} &= \hat{p_\text{U}} - \hat{p_\text{V}} = \frac{y_U}{n_U} - \frac{y_V}{n_V} \\
    \hat{\sigma^2} &= \frac{\hat{p_U}(1-\hat{p_U})}{n_U} + \frac{\hat{p_V}(1-\hat{p_V})}{n_V} \tag{16}

Plugging these two values into the equations from the previous section will
give us a good approximation of the lift in terms of the activation rate.

|h2| Selecting a Sample Size |h2e|

The above sections is about finding a confidence interval *after* you have all
your observations.  What if you want to ensure that you have statistically
significance results?  The only thing you can (usually) do a priori is pick the
sample size.

There are two main ways to select a sample size: (i) using an error bound, and (ii) using
the hypothesis testing framework.  Let's take a look at both.


|h3| Sample Size using an Error Bound |h3e|

In this method, we'll be using our confidence interval from Equation 13.
We can see that our true mean is bounded within 
:math:`\pm z_{\alpha/2}\sqrt{\frac{\sigma^2_U}{n_U} + \frac{\sigma^2_V}{n_V}}` our esimate of lift.
Limiting this quantity to a specific value (:math:`B`) and solving for 
:math:`n`, we can compute our desired sample size.
(Set :math:`n = n_U = c n_V` to make our computation a bit simpler.)

.. math::

    z_{\alpha/2}\sqrt{(\frac{\sigma^2_U}{n} + \frac{\sigma^2_V}{cn})} &= B \\
    \sqrt{\frac{\sigma^2_U}{n} + \frac{\frac{\sigma^2_V}{c}}{n}} &= \frac{B}{z_{\alpha/2}} \\
    \sqrt{\frac{n}{\sigma^2_U + \frac{\sigma^2_V}{c}}} &= \frac{z_{\alpha/2}}{B}\\
    n &= \frac{z_{\alpha/2}^2}{B^2} (\sigma^2_U + \frac{\sigma^2_V}{c}) \tag{17}

The only caveat here is finding an estimate for :math:`\sigma` (remember we're
doing this before we have any observations), so we can't use any samples to
estimate it.  In that case, you could use a previously known sample variance 
(from a similar experiment) or another quick and dirty estimate is that
the range of allowable values usually falls within :math:`4\sigma`.
Both these will provide a decent estimate of the values.
Let's take a look at an example.


.. admonition:: Example 1

 From a past experiment, we know that customers usually spend between
 $20 and $140 during a promotion period. We can send
 :math:`20,000` customers flyers and want an error bound on spend per customer
 of $0.50 with 95% confidence.  How many people should we allocate for
 treatment and control?
 
 First, find an approximate standard deviation:

 .. math::

    \text{range} &= (80-20) = 60  \approx 4\sigma \\
    \sigma   &\approx 15 \tag{18}

 Using this estimate (assuming that it is valid for both control and treatment)
 and Equation 17 (with :math:`1 - \alpha = 0.95` so :math:`z_{0.05/2}\approx 1.96`), we get:

 .. math::

    n &= \frac{z^2_{\alpha/2}}{B^2} (\sigma^2_U + \frac{\sigma^2_V}{c}) \\
      &= \frac{1.96^2}{0.5^2}(1 + \frac{1}{c})(15^2) \\
      &= (1 + \frac{1}{c})(3457.44) \tag{19}

 We want to allocate as small a control group as possible so we can
 maximize revenue (assuming our promotion has positive lift).  Knowing that
 :math:`n(1 + c) = 20,000` (since treatment and control add up to this number),
 solving for :math:`n` with Equation 19 (using the quadratic equation):

 .. math::

    n = (1 + \frac{1}{c})(3457.44) &= \frac{20000}{1 + c} \\
        (c+1)^2 &= \frac{20000}{3457.44}c \\
        c^2 - 3.7846c + 1 = 0 \\
        c \approx 3.4988 \pm 0.28581 \\
        n \approx 4446 \text{ or } 15554 \tag{20}

 The two solutions correspond to :math:`n` being the larger or smaller number
 because we make not assumptions about which one is larger (:math:`n` or
 :math:`cn`).  Thus, we should pick the treatment group to be approximately 15550
 and control to be 4450.  Contrast this with setting :math:`c=1` in Equation
 19, which would yield :math:`n=6915`, a slightly larger control group than is
 necessary.


|h3| Sample Size using Hypothesis Testing and Statistical Power |h3e|

Another method to pick sample size is to use a hypothesis testing
framework along with statistical power.  To conduct this procedure, we need a
few things:

* :math:`\alpha`: the false positive rate (or how often we incorrectly detect
  something is true when it's not).  This is usually set a :math:`0.01` or
  :math:`0.05` in most scientific experiments.
* Power (denoted by :math:`1 - \beta`, where :math:`\beta` is the false
  negative rate): How often we are able to conclude that the alternative
  hypothesis is true when it is.  A common value that is used is usually :math:`0.80`.
* Minimum detectable effect size (denoted by :math:`\Delta`): The minimum effect size (i.e. lift) we want be able
  to detect.  For example, we may choose a $0.50 SPC as the minimum detectable effect size.

The basic idea is first we establish a test or "rule" using our the
hypothesis testing framework (and a given :math:`\alpha`) to decide
when we accept and when we reject a given sample.
Next, we use this rule along with the power constraint and the minimum
detectable effect size to compute the required :math:`n`. 
Let's take a look in detail.

First, let's determine what our test is for determining if something
is statistically significant.  Since we're dealing with large sample sizes,
we've already determined that we're working with normally distributed variables.
The uniformly most powerful test in this case (which you can derive using the 
`Neyman-Pearson Lemma <https://en.wikipedia.org/wiki/Neyman%E2%80%93Pearson_lemma>`_)
is given by:

.. math::

    P(\bar{y} > \mu_0 + \frac{z_{\alpha/2}\sigma}{\sqrt{n}} | H_0) = \alpha \tag{21}

Translating that into our problem with the outcome variable being lift,
the null hypothesis that the lift is zero, we get:

.. math::

    P(\text{lift} = \bar{U} - \bar{V} > \frac{z_{\alpha/2}\sigma_{\text{lift}}}{\sqrt{n}} | \mu_{\text{lift}} = 0) = \alpha \tag{22}

Now that we have established our test: 
:math:`\bar{U} - \bar{V} > \frac{z_{\alpha/2}\sigma}{\sqrt{n}}`, we can 
see how often we will correctly identify the alternative hypothesis to be true:

.. math::

    P(\text{lift} &> \frac{z_{\alpha/2}\sigma_{\text{lift}}}{\sqrt{n}} | \mu_{\text{lift}} = \Delta) \geq 1 - \beta \\
    P(\frac{\text{lift} - \Delta}{\sigma_{\text{lift}} / \sqrt{n}} &> (\frac{z_{\alpha/2}\sigma_{\text{lift}}}{\sqrt{n}} - \Delta)\frac{1}{\sigma_{\text{lift}} / \sqrt{n}} | \mu_{\text{lift}} = \Delta) = 1 - \beta \\
    P(Z &> z_{\alpha/2} - \frac{\sqrt{n}\Delta}{\sigma_{\text{lift}}} | \mu_{\text{lift}} = \Delta) \geq 1 - \beta   && \text{since lift is normally distributed} \\
    -z_{\alpha/2} - \frac{\sqrt{n}\Delta}{\sigma_{\text{lift}}} &\geq z_{1 - \beta} \\
    \frac{\sqrt{n}\Delta}{\sigma_{\text{lift}}} &\geq z_{1 - \beta} + z_{\alpha/2} \\
    n &\geq \sigma^2_{\text{lift}} \frac{(z_{1 - \beta} + z_{\alpha/2})^2}{\Delta^2} \tag{23}

Let's take a look at an example of how this works.

.. admonition:: Example 2

  Continuing from Example 1, how large of a sample size do we require if :math:`1-\beta=0.80` but now we
  can send :math:`30,000` flyers?

  Using Equation 23 and estimating :math:`\sigma \approx 15`, we have (where :math:`z_{1-\beta}=0.84`):

  .. math::

    n &\geq (\sigma^2_U + \frac{\sigma^2_V}{c}) \frac{(z_{1 - \beta} + z_{\alpha/2})^2}{\Delta^2} \\
      &= (1 + \frac{1}{c})(15)^2 \frac{0.84 + 1.96)^2}{0.5^2}  \\
      &= 7056 (1 + \frac{1}{c}) \tag{24}
 
  With the added constraint :math:`n(1 + c) = 30000`, we solve for :math:`c`:

  .. math::

    n = (1 + \frac{1}{c})(7056) &= \frac{30000}{1 + c} \\
        (c+1)^2 &= \frac{30000}{7056}c \\
        c^2 - 2.252c + 1 = 0 \\
        c \approx 1.644 \pm 0.6084 \\
        n \approx 11349 \text{ or } 18651 \tag{25}

  So we should send approximately 11350 flyers to the control group and 18650 to the treatment group.
  Contrast this with setting :math:`c=1` resulting in :math:`n=14112`.

|h3| Binomial Outcome Variables |h3e|

One last point, the above calculations for sample size are equally valid when the outcome
is binary (e.g. for conversion or activation rate).  The big difference is how we estimate
the standard deviation/variance.  Since we're dealing with a binary outcome, we can model
the total number of customers who convert as a binomial random variable
(:math:`Y`), meaning each customer can be modeled as a Bernoulli random
variable (:math:`X`), both with underlying parameter :math:`p`.
Using the same line of reasoning above, a good estimate for the standard deviation
of :math:`\sigma_X` uses variance of a Binomial random variable:

.. math::

    \sigma^2 = \frac{np(1-p)}{n} = p(1-p) \tag{26}

With Equation 26, to estimate :math:`\sigma_U` or :math:`\sigma_U` in Equation 23,
we need to estimate :math:`p`.  Usually this can be estimated based on prior campaign
that you ran where you have a ballpark of the previous conversion rate.
Putting the two together with estimate :math:`\hat{p}`:

.. math::

    n \geq (\hat{p}(1-\hat{p}) + \frac{\hat{p}(1-\hat{p})}{c}) \frac{(z_{1 - \beta} + z_{\alpha/2})^2}{\Delta^2} \tag{27}

For example, you might expect the baseline conversion rate of the control group
to be approximately 5% (:math:`\hat{p}=0.05`).  In that case, we can easily
solve for :math:`n` as before.



|h2| Conclusion |h2e|

Whew!  This post was a lot longer than I expected.  "Elementary" is such a misleading
word because in some cases it's obvious and others exceedingly complex.  The
reason why statistics is sometimes inaccessible is the derivations and details
of "elementary" statistics is sometimes a bit complex (even though the actual
procedure is simple).  Hopefully this primer will help put both direct
marketing and elementary statistics in perspective while giving some intuition
on both subjects.


|h2| References and Further Reading |h2e|

* Wikipedia: `Direct marketing <https://en.wikipedia.org/wiki/Direct_marketing>`_, `Central Limit Theorem <https://en.wikipedia.org/wiki/Central_limit_theorem>`_, `Sample Mean <https://en.wikipedia.org/wiki/Sample_mean_and_covariance>`_, `Sample Variance <https://en.wikipedia.org/wiki/Variance#Sample_variance>`_, `Law of Large Numbers <https://en.wikipedia.org/wiki/Law_of_large_numbers>`_.
* `All of Statistics: A Concise Course in Statistical Inference <http://link.springer.com/book/10.1007%2F978-0-387-21736-9>`_ by Wasserman.
* `Mathematical Statistics with Applications <http://www.amazon.ca/Mathematical-Statistics-Applications-Dennis-Wackerly/dp/0495110817>`_ by Wackerly, Mendenhall and Scheaffer.
* `Data Mining Techniques: For Marketing, Sales, and Customer Relationship Management <http://www.amazon.com/Data-Mining-Techniques-Relationship-Management/dp/0470650931/>`_ by Linoff.
* `How Not To Run An A/B Test <http://www.evanmiller.org/how-not-to-run-an-ab-test.html>`_, Evan Miller.
  

|br|
|br|


.. [1] An interesting story I read about in `The Emperor of All Maladies <https://en.wikipedia.org/wiki/The_Emperor_of_All_Maladies>`_ about how several large scale trials testing the effectiveness of mammograms (low-dose x-ray imaging to detect breast cancer) were undone by implicit selection biases.  In one of the Canadian studies, nurses would write down trial patients names in a notebook where the first line corresponded to the treatment group, second control group, third treatment and so forth.  The nurses administering the trial subtlely biased the results by feeding patients who they thought were more in need to the treatment group.  A compassionate gesture but, statistically, a failed experiment.  Without the benefit of a truly randomized trial, they could not longer analyze the effectiveness of the treatment in insolation of confounding variables.

.. [2] The topic of design of `randomized control trials <https://en.wikipedia.org/wiki/Randomized_controlled_trial>`_ can is actually be quite complex.  In this explanation, we're assuming relatively large population sizes (10s of thousands), which allows us to make a lot of simplifying assumptions.  When dealing with small sample sizes, the randomness of the samples can be quite large making it much more important to design your experiment properly so you can get proper conclusions.  For larger population sizes, we usually worry less about this.

.. [3] Many marketers are tempted to *not* take a control group.  Their reasoning is something along the lines of "but I'm missing out on sales!", in which you should respond back "how do you know that?"  It's quite possible the offer could have absolutely no meaningful effect on your customers (e.g. targeting wrong product to people who don't want it) and possibly a negative effect (e.g. the discount may be too high with not enough people coming in)!  Just because you're giving people a discount doesn't mean it always increases sales.  Further, even if you have an increase, you don't know *how much* of an increase it is by i.e. the effect size.  If you different offers boosted sales by 1% vs. 10%, you should know that!  This is how you can test and learn to improve your overall business.

.. [4] For example, :math:`X` and :math:`Y` represent random variables which we can manipulate and analyze properties from *before* we have actualy observed any values for them.  That means any analysis we apply on them doesn't depend one the actual numbers we observe.  After we observe some samples, we'll have explicit values for them like :math:`x=1` and :math:`y=48`, where we use lower case to distinguish these realizations of the random variables.

.. [5] This section was primarily based on *All of Statistics*, Chapter 16.  It has a great explanation of how randomized control groups work.  Check it out if my quick explanation glosses over too many things.


