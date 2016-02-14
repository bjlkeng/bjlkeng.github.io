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

To not bury the lead, a good point estimator for measuring campaign
effectivness is just using the difference of spend per customer (SPC) or
**lift**  of the treatment and control group (i.e. different of population
means):

.. math::

    \hat{\text{lift}} &= \hat{E}[Y|X=1] - \hat{E}[Y|X=0] \\
                      &= \bar{Y}_{X=1} - \bar{Y}_{X=0} \\
                      &= \frac{1}{n_T} \Sigma_{i=1}^{n} Y_i X_i
                         - \frac{1}{n_C} \Sigma_{i=1}^{n} Y_i (1 - X_i) \\
                      &= SPC_{\text{treatment}} - SPC_{\text{control}}          \tag{2}

where the :math:`\hat{}` symbol represents an estimate,
:math:`n_T = \Sigma_{i=1}^{n} X_i` and :math:`n_C = \Sigma_{i=1}^{n} (1-X_i)`.
All those equations boil down to basically just taking the difference of spend
per customer between treatment and control.  In some of the more math heavy
sections below, you'll see why we introduced all this notation.
First, let's see why our control groups have to be a random sample.

|h3| Causal Inference [4]_ |h3e|

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

.. admonition:: **Theorem** 

 If we randomly assign subjects to treatment and control such that :math:`P(X=0) > 0`
 and :math:`P(X=1) > 1`, then :math:`\alpha=\text{lift}`.
    
 **Proof**
     Since X is randomly assigned, X is independent of :math:`C_{X=1}, C_{X=0}`, so:
 
     .. math::
     
         \text{lift} &= E(C_{X=1}) - E(C_{X=0}) \\
                     &= E(C_{X=1}|X=1) - E(C_{X=0}|X=0) && \text{since } X \text{ is independent of } C_X \\
                     &= E(Y|X=1) - E(Y|X=0) && \text{since } Y = C_X \\
                     &= \alpha   \tag{7}


Using Theorem 1 we can see that by assigning random control groups, our
use of difference in SPC (i.e. association) using in Equation 2 is identical
the actual causal effect that we want, which is lift.  However, if we
don't have random assignments then there is no guarantee that the association
we computed in Equation 2 has anything to do with lift, as we saw in Example 1.


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
we can see that the sample mean of each population (treatment :math:`\bar{U}` and control :math:`\bar{V}`) can
be approximated by a normal distribution, whose expected value is the mean of
the underlying variable:
 
.. math::

    E[\bar{U}] &= E[\frac{1}{n_U} \Sigma_{i=1}^{n_U} U_i] \approx E[N(\mu_U, \frac{\sigma^2_U}{n_U})] = \mu_U \\
    E[\bar{V}] &= E[\frac{1}{n_V} \Sigma_{i=1}^{n_V} V_i] \approx E[N(\mu_V, \frac{\sigma^2_V}{n_V})] = \mu_V \tag{8}

Equation 8 also confirms our estimators are consistent because
:math:`E[\bar{U}] \approx \mu_U` and :math:`E[\bar{V}] \approx \mu_V` for large :math:`n`.
Usually our :math:`n` is quite large (:math:`>10000`) so our normal
approximations are quite good.
Similarly, using the strong `law of large numbers <https://en.wikipedia.org/wiki/Law_of_large_numbers>`_,
the `sample variance <https://en.wikipedia.org/wiki/Variance#Sample_variance>`_
(denoted by :math:`s^2`) is a pretty good approximation of the actual variance
when we have large :math:`n`:

.. math::

    s_U^2 &\approx \sigma_U^2 \\
    s_V^2 &\approx \sigma_V^2 \tag{9}

Using Equation 8 and 9, we get our approximation of the sample mean (and spend
per customer) for large :math:`n` (the :math:`\hat{}` denote our actual point
estimates for the quantities):
    
.. math::

    SPC_T &\approx N(\hat{\bar{U}}, \frac{\hat{s}^2_U}{n_U}) \\
    SPC_C &\approx N(\hat{\bar{V}}, \frac{\hat{s}^2_V}{n_V})  \tag{10}

Knowing that the `difference of two normal distributions
<http://mathworld.wolfram.com/NormalDifferenceDistribution.html>`_ is just a
normal distribution (with mean equal to the difference of the means, and
variance equal to the sum of variances), our lift is:


.. math::

     \text{lift} &= SPC_T - SPC_C \\
                 &\approx N(\bar{U}, \frac{s^2_U}{n_U}) - N(\bar{V}, \frac{s^2_V}{n_V}) \\
                 &= N(\bar{U} - \bar{V}, \frac{s^2_U}{n_U} + \frac{s^2_V}{n_V}) \tag{11} 

Now our lift is simply just a normal random variable whose mean and variance we know how to estimate,
we can get a :math:`1 - \alpha` two sided confidence interval.  Since lift is approximately normal,
we know that :math:`Z=\frac{\hat{\text{lift}} - \mu_{\text{lift}}}{\sigma_{\text{lift}}} \approx \frac{\hat{\text{lift}} - \mu_{\text{lift}}}{\hat{s}_{\text{lift}}}` 
has a standard normal distribution N(0, 1):

.. math::

    P(-z_{\alpha/2} &\leq Z \leq Z_{\alpha/2}) &= 1 - \alpha \\
    P(-z_{\alpha/2} &\leq \frac{\hat{\text{lift}} - \mu_{\text{lift}}}{\hat{s}_{\text{lift}}} \leq z_{\alpha/2}) = 1 - \alpha \\
    P(-z_{\alpha/2} \hat{s}_{\text{lift}} &\leq \hat{\text{lift}} - \mu_{\text{lift}} \leq z_{\alpha/2} \hat{s}_{\text{lift}}) = 1 - \alpha \\
    P(-z_{\alpha/2}{\hat{s}_{\text{lift}}} - \hat{\text{lift}} &\leq - \mu_{\text{lift}} \leq z_{\alpha/2}{\hat{s}_{\text{lift}}} - \hat{\text{lift}}) = 1 - \alpha \\
    P(\hat{\text{lift}} - z_{\alpha/2}{\hat{s}_{\text{lift}}} &\leq \mu_{\text{lift}} \leq \hat{\text{lift}} + z_{\alpha/2}{\hat{s}_{\text{lift}}}) = 1 - \alpha \tag{12}

That is, the true lift (:math:`\mu_{\text{lift}}`) lies in the interval :math:`[\hat{\text{lift}} - z_{\alpha/2}{\hat{s}_{\text{lift}}}, \hat{\text{lift}} - z_{\alpha/2}{\hat{s}_{\text{lift}}}]`,
:math:`1-\alpha` of the time, where :math:`\hat{\text{lift}}` is our point
estimate of lift and :math:`z_{\alpha/2}` is the :math:`\alpha/2` z-score for a standard normal distribution.

|h3| Activation Rate and Binomial Outcome Variables |h3e|

All the math above equally applies to when your outcome variable is not a
real-valued number such as a activation or conversion rate.  In that case, each
customer can only have two outcomes: :math:`1` (shops or converts) and
:math:`0` (doesn't shop or convert).  There are a few caveats though.

The standard unbiased estimators for mean and variance of a binomial would be
used where :math:`Y` are the number of successes (or :math:`1`'s) in the
sample, and :math:`n` are the total number of samples:

.. math::

    \hat{\mu} &= \frac{Y}{n} \\
    \hat{\sigma^2} &= \frac{\hat{p}(1-\hat{p})}{n} \tag{13}

Plugging these two values in the above equation will give us a good
approximation of the true lift in terms of the activation rate.

A good rule of thumb of when you can use a normal to approximate a binomial
is (from Wackerly et.  al):

.. math::

    n \geq 9 \frac{\text{larger of }p\text{ and }(1-p)}{\text{smaller of }p\text{ and }(1-p)} \tag{14}

So for :math:`p=0.01`, :math:`n \geq 9 \frac{0.99}{0.01} = 891`, meaning you
want your treatment and control groups to be at least 900.  Most likely you
will want bigger values of :math:`n` to control the error rate (see below).

|h3| Statistical Power and Selecting a Sample Size |h3e|




|h2| Conclusion |h2e|

|h2| References and Further Reading |h2e|

* Wikipedia: `Direct marketing <https://en.wikipedia.org/wiki/Direct_marketing>`_, `Central Limit Theorem <https://en.wikipedia.org/wiki/Central_limit_theorem>`_, `Sample Mean <https://en.wikipedia.org/wiki/Sample_mean_and_covariance>`_, `Sample Variance <https://en.wikipedia.org/wiki/Variance#Sample_variance>`_, `Law of Large Numbers <https://en.wikipedia.org/wiki/Law_of_large_numbers>`_.
* `All of Statistics: A Concise Course in Statistical Inference <http://link.springer.com/book/10.1007%2F978-0-387-21736-9>`_ by Wasserman.
* `Mathematical Statistics with Applicatinos <http://www.amazon.ca/Mathematical-Statistics-Applications-Dennis-Wackerly/dp/0495110817>`_ by Wackerly, Mendenhall and Scheaffer.
* `Data Mining Techniques: For Marketing, Sales, and Customer Relationship Management <http://www.amazon.com/Data-Mining-Techniques-Relationship-Management/dp/0470650931/>`_ by Linoff.
  
  

|br|
|br|


.. [1] An interesting story I read about in `The Emperor of All Maladies <https://en.wikipedia.org/wiki/The_Emperor_of_All_Maladies>`_ about how several large scale trials testing the effectiveness of mammograms (low-dose x-ray imaging to detect breast cancer) were undone by implicit selection biases.  In one of the Canadian studies, nurses would write down trial patients names in a notebook where the first line corresponded to the treatment group, second control group, third treatment and so forth.  The nurses administering the trial subtlely biased the results by feeding patients who they thought were more in need to the treatment group.  A compassionate gesture but, statistically, a failed experiment.  Without the benefit of a truly randomized trial, they could not longer analyze the effectiveness of the treatment in insolation of confounding variables.

.. [2] The topic of design of `randomized control trials <https://en.wikipedia.org/wiki/Randomized_controlled_trial>`_ can is actually be quite complex.  In this explanation, we're assuming relatively large population sizes (10s of thousands), which allows us to make a lot of simplifying assumptions.  When dealing with small sample sizes, the randomness of the samples can be quite large making it much more important to design your experiment properly so you can get proper conclusions.  For larger population sizes, we usually worry less about this.

.. [3] Many marketers are tempted to *not* take a control group.  Their reasoning is something along the lines of "but I'm missing out on sales!", in which you should respond back "how do you know that?"  It's quite possible the offer could have absolutely no meaningful effect on your customers (e.g. targeting wrong product to people who don't want it) and possibly a negative effect (e.g. the discount may be too high with not enough people coming in)!  Just because you're giving people a discount doesn't mean it always increases sales.  Further, even if you have an increase, you don't know *how much* of an increase it is by i.e. the effect size.  If you different offers boosted sales by 1% vs. 10%, you should know that!  This is how you can test and learn to improve your overall business.

.. [4] This section was primarily based on *All of Statistics*, Chapter 16.  It has a great explanation of how randomized control groups work.  Check it out if my quick explanation glosses over too many things.

.. [5] This section was primarily based on *All of Statistics*, Chapter 16.  It has a great explanation of how randomized control groups work.  Check it out if my quick explanation glosses over too many things.
