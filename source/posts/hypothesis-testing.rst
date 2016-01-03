.. title: A Primer on Statistical Inference and Hypothesis Testing
.. slug: hypothesis-testing
.. date: 2015-12-29 10:22:26 UTC-05:00
.. tags: hypothesis testing, frequentist statistics, statistical inference, models, p-values, mathjax
.. category: 
.. link: 
.. description: A post explaining classical (frequentist) statistical inference and hypothesis in a (hopefully) straight-forward way.
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

This post is about some fundamental concepts in classical (or frequentist)
statistics: inference and hypothesis testing.  A while back, I came to the
realization that I didn't have a rigorous understanding of these concepts (at least
not to my liking) beyond the mechanical nature of applying them.
This led to a sense of intuition on how these techniques relate to the more
fundamental area of probability.  This bothered me a lot since having a good
intuition about a subject is probably the most useful (and fun!) part of learning
a subject.  So this post is a result of my re-education on these topics.
Enjoy!

.. TEASER_END

|h2| Statistical Models and Inference |h2e|

|h3| A Couple of Big Ideas |h3e|

To start from the beginnging, there are two big ideas that underlie much of 
classical statistics.
The first big idea is that *all data* (or observations as statisticians like to
say) have a "true" probability distribution [1]_.  Of course, it is almost never
possible to precisely define it because the real world rarely fits so nicely
into the distributions we learn in stats class.
The implications of this idea is that the "true" distribution and its
parameters are fixed (i.e.  *not* random) albeit unknown.  The randomness
comes in when you sample from this "true" distribution from which each datum
is randomly drawn.

The second big idea is that **statistical inference** [2]_ (or as computer
scientists call it "learning" [3]_) basically boils down to estimating this
distribution directly by computing the distribution or density function [4]_,
or indirectly by estimating derived metrics such as the mean or median of the
distribution.  A typical question we might ask is:

    Give a sample :math:`X_1, X_2, \ldots, X_n` drawn from a distribution
    :math:`F`, how do we estimate :math:`F` (or some properties of :math:`F`)?

Of course there are variations to this question depending on the precise
problem such as regression but by and large it comes down to finding things
about :math:`F` (or its derived properties).

|h3| Models, models, models |h3e|

Now that we have those two big ideas out of the way, let's define a
(statistical) model:

    A **statistical model** :math:`\mathfrak{F}` is a set of distributions (or
    densities or regression functions).

The idea here is that we want to define a subset of all possible distributions
(or densities or regression functions) that closely approximates the "true"
distribution (whether or not :math:`\mathfrak{F}` actually contains :math:`F`
[5]_).  One of the first tasks in inferential procedures is selecting the
correct model.  The model is an *assumption* about your data, picking the wrong
one will lead to invalid conclusions.

By far, the most common type of model is a **parametric model**, which defines
:math:`\mathfrak{F}` using a finite number of parameters.  For example, if we
assume that the data comes from a Normal distribution, we would use the
parametric model for a Normal distribution:

.. math::

  \mathfrak{F} = \big\{ f(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}}
  e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \mu \in \mathbb{R}, \sigma > 0 \big\}
  \tag{1}

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
Another example would be a k-nearest neighbour algorithm that can classify a new
observation solely based on its k-nearest neighbours from training data.  The
surface defined by the classification function is not pre-defined rather it is
determined solely by the training data (and hyper parameter :math:`k`).  You can
contrast this with a logistic regression as a classifier, which has a rigid
structure regardless of how well the data matches. 

Although, it sounds appealing to let the "data define the model",
non-parametric models typically requires a much larger sample size to draw a
similar conclusion compared to parametric methods.  This makes sense
intuitively since parametric methods have the advantage of having the extra
model assumptions, so making conclusions should be easier all else being equal.
Of course, you must be careful picking the *right* parametric model or else
it will lead you to incorrect conclusions.

|h3| Types of Statistical Inference |h3e|

For the most part, statistical inference problems can be broken into three
different types of problems [6]_: point estimation, confidence
intervals, and hypothesis testing.  I'll briefly describe the former two
and focus on the latter in the next section.

Point estimates aims to find the single "best guess" for a particular quantity
of interest.  The quantity could be the parameter of a model, a CDF/PDF, or a
regression/prediction function.  Formally:

    For :math:`n` independent and identically distributed (IID)
    observations, :math:`X_1, \ldots, X_n`, from some distribution :math:`F` with
    parameter(s) :math:`\theta`, a **point estimator** :math:`\widehat{\theta}_n`
    of parameter :math:`\theta` is some function of :math:`X_1, \ldots, X_n`:

    .. math::
    
      \widehat{\theta}_n = g(X_1, \ldots, X_n). \tag{2}

For example, if our desired quantity is the expected value of the "true"
distribution :math:`F`, we might use the sample mean of our data as our "best
guess" (or estimate).  Similarly, for a regression problem with a linear model, we are
finding a "point" estimate for the regression function :math:`r`, which is
frequently the coefficients for the covariates (or features) that minimize the
mean squared error.  From what I've seen, many "machine learning" techniques
fall in this category where you typically will aim to find a maximum likelihood
estimate or related measure that is your "best guess" (or estimate) based on
the data.


The next category of inference problems are confidence intervals (or sets).
The basic idea here is that instead of finding a single "best guess" for a
parameter, we try to find an interval that "traps" the actual value of the
parameter (remember the observations have a "true" distribution) with a
particular frequency.  Let's take a look at the formal definition then try to
interpret it:  

    A :math:`1-\alpha` **confidence interval** for parameter
    :math:`\theta` is an interval :math:`C_n(a,b)` where :math:`a=a(X_1, \ldots, X_N)` 
    and :math:`b=b(X_1, \ldots, X_N)` are functions such that 

    .. math::
    
        P(\theta \in C_n) \geq 1 - \alpha. \tag{3}

This basically says that our interval :math:`(a,b)` "traps" the true value of
:math:`\theta` with probability :math:`1 - \alpha` .  Now the confusing part is
that this does not say anything directly about the probability of
:math:`\theta` occurring because :math:`\theta` is fixed (from the "true"
distribution) and instead it is :math:`C_n` that is the random variable [7]_.
So this is more a statement about how "right" we were in picking :math:`C_n`.

Another way to think about it is this: suppose we always set :math:`\alpha = 0.05` 
(a 95% confidence interval) for every confidence interval we ever compute,
which will be composed of any variety of different "true" distribution and
observations.  We would expect that the respective :math:`\theta` in each case
to be "trapped" in our confidence interval 95% of the time.  Note this is
different from saying that on any one experiment we "trapped" :math:`\theta`
with a 95% probability -- after we have a realized confidence interval (i.e.
fixed values), the "true" parameter either lies in it or it doesn't.

In some ways confidence intervals give us more context then a single point
estimate.  For example, if we're looking at the response of a marketing campaign
versus a control group, the difference in response or  *incremental lift* is a
key performance indicator.  We could just compute the difference in the sample
mean of the two populations to get a point estimate for the lift, which might
show a positive result say 1%.  However, if we computed a 95% confidence
interval we might get :math:`(-0.015, 0.0155)` which overlaps with 0, implying
that our 1% lift may not be statistically significant.

Conceptually, point estimates and confidence intervals are not *that* hard to
understand.  The complexity comes in when you have to actually pick an
estimator that has nice properties (like minimizing bias and variance) in the
case of Equation 2, or picking an interval such that Equation 3 is satisfied.
Thankfully, many smart mathematicians and statisticians have figured out
estimators and confidence intervals for many common situations so we rarely
need to derive things from scratch.  Instead, we can pick the most appropriate
technique for the problem at hand.

|h2| Hypothesis Testing |h2e|

|h3| A Digression |h3e|

I'm a huge fan of hypothesis testing as a general concept (not necessarily
statistical) because it's such a powerful framework for learning.  One of the
biggest advantages is it sets you up to "disprove your best-loved ideas" as
Charlie Munger puts it, not to mention the hundreds of years
its been used as part of the `scientific method <https://en.wikipedia.org/wiki/Scientific_method>`_.
There is a huge advantage to having a mental framework that allows you to
disprove your hardest won ideas, a proverbial `"empty your cup" <http://c2.com/cgi/wiki?EmptyYourCup>`_ type situation where
you can begin to learn after you have let go of your some of your past
(hopefully, incorrect) beliefs.  I mean that's what science is all about right?

Before we conclude this interlude, I want to mention two important points that
stood out when practicing hypothesis testing.  The first idea is that of testability or
falsifiability.  This is rather important because if you have a hypothesis that
cannot be falsified or not have a reasonable expectation of observing a
counter-example then it's not a very useful hypothesis.  That is, you won't
really be able to learn much from it because you can never know if it's false [8]_.

The second idea is that of parsimony, or `Occam's Razor <https://en.wikipedia.org/wiki/Occam%27s_razor>`_.
In short, we should prefer the simplest explanation or model, which translates
to hypotheses that have the fewest assumptions.  It makes sense not only from a
machine learning point of view (over-fitting) but also from an intuitive point
of view (more assumptions results in a weaker explanation).  A related idea that 
Charlie Munger espouses is that when explaining things, we should favour the
explanation from the most fundamental discipline.  For example, I'm sure many
economists have some fancy names for the reasons behind the financial crisis
but we can probably use simpler terminology and concepts related to breaking
points and critical mass from engineering and physics.  On an individual level,
it makes more sense to pull ideas and concepts from the most fundamental of
disciplines (math, physics, engineering) because they are the most reliable.
Anyways, that's enough of a digression, back to statistics!


|h3| Statistical Hypothesis Testing |h3e|

Statistical hypothesis testing is probably one of the earliest concepts
learn in a statistics course.  Null hypotheses, Student's t-test, p-values
these terms get thrown around a lot without explaining their underlying probabilistic
basis [9]_.  When I first learned statistics it was definitely more biased towards
a mechanical view of hypothesis testing, rather than an intuitive understanding.
Here's my attempt to explain it a bit more precisely while hopefully adding some
colour to give some intuition.

Following the scientific method, we make a hypothesis, run an experiment and
see if our observations match the prediction from our hypothesis.  However
in certain cases, the cause and effect is not so clear like it is with laws of
nature.  For example, when you `double-slit experiment <https://en.wikipedia.org/wiki/Double-slit_experiment>`_ 
to determine the dual nature of light, the result of the experiment is clear.
But when you're determining if a new drug helps cure a disease, you usually 
randomly divide a population into a treatment group which gets the drug and a
control group which receives a placebo.  If we look at the various scenarios 
of what can happen, we can see why it's not so clear cut:

1. If at least one person in the treatment group doesn't get better, does it
   mean the drug isn't effective? 
  
   Not necessarily, the drug could still be
   quite effective but for some other *random* reason, the person could have
   not responded to the drug by pure chance.

2. If more people in the treatment group get better than the control, does it
   mean the drug is effective?  
   
   Not necessarily, what if only the treatment group has only 1 person who got
   better versus control. In this case, probably not, it could be due to
   another random factor.  10? 1000?  Maybe.

You can start to see why we need to apply some mathematics to these
situations in order to see if the effect is significant.  In particular, we
apply statistical hypothesis testing when we want to determine if an observed
effect is really there or just happening by purely chance (i.e. other random
factors).

The high level setup for this procedure is to first come up with a null
hypothesis (denoted by :math:`H_0`), that usually denotes the "no effect"
scenario, or our default position.  We then try to see how likely the data
is generated in this situation.  If it's unlikely then we say we *reject* the
null hypothesis and accept the alternate hypothesis, which just means something
other than the null hypothesis must true.  Otherwise, we have no evidence to
reject the null hypothesis and we continue to believe it to be true (since it's
our default position).  

A good analogy is that of a legal trial, the defendant is innocent until proven
guilty.  Likewise, we assume the null hypothesis is true from the start, and only
when we reject it do we say it is false.  This is not unlike how science works
where we have established models that are assumed to be true until later proven
otherwise.  Now that we have a conceptual understanding of this process, let's look
at some details.

|h3| Rejection Regions and Types of Errors |h3e|

A critical point when conducting statistical hypothesis testing is
determining your null hypothesis.  The first step of this process is picking an *appropriate*
statistical model :math:`\mathfrak{F}`.  If your model is ill-formed for your
problem, the results of hypothesis testing will be invalid.  Next, we partition
the parameter space of :math:`\mathfrak{F}` into two disjoint sets :math:`\Theta_0`
and :math:`\Theta_1`, and define our hypotheses as:

.. math::

    H_0 : \theta \in \Theta_0 \\
    H_1 : \theta \in \Theta_1 \tag{4}

where :math:`H_0` is our **null hypothesis** and :math:`H_1` is our
**alternative hypothesis**.  So we must first pick a good statistical model
then define an appropriate null hypothesis.  For example, we might pick a
normal distribution as our statistical model and our null hypothesis is that
the mean of the distribution is less than or equal to zero (:math:`\mu \leq 0`) .

Now let's suppose our data are represented by the random variable :math:`X`
with range :math:`\chi` (all possible values of :math:`X`).  Our goal is to
define a **rejection region** on :math:`\chi` such that:

.. math::

   X \in R    &\implies \text{ reject } H_0 \\
   X \notin R &\implies \text{ retain (don't reject) } H_0 \tag{5}

We want to define :math:`R` such that when :math:`H_0` is true, we have a high
probability of retaining :math:`H_0` and when :math:`H_0` is false, we have a
high chance probability of rejecting it.  If we picked a good :math:`R`, when
we actually observe our data then it should be quite simple to check if
:math:`X \in R` and be correct quite often.

Another way to view this is in terms of the errors we could make.  
If we reject :math:`H_0` when it's actually true, we've committed a **Type
I Error** or false positive (whose probability is denoted by :math:`\alpha`).  If we retain
:math:`H_0` when it's actually false, we've committed a **Type II Error** or false
negative (whose probability is denoted by :math:`\beta`).  Here's a summary:

|center|

================= ============================= =============
Cases             Retain Null                   Reject Null
================= ============================= =============
:math:`H_0` true  Correct                       Type I Error (:math:`\alpha`)
----------------- ----------------------------- -------------
:math:`H_0` false Type II Error (:math:`\beta`) Correct
================= ============================= =============

|centere|

So it makes sense that we want choose :math:`R` to maximize the "Correct"
diagonals or alternatively minimize "Error" diagonals in the above table.  To
throw another wrench in the mix, we usually refer to the bottom right cell
as the **power**, which is the probability of correctly rejecting the null
hypothesis when it is false (i.e. the alternative hypothesis is true).

The tricky part is that trying to minimize both :math:`\alpha` and
:math:`\beta`, which results in conflicting goals [10]_.  So picking a good rejection region :math:`R` is
non-trivial.

|h3| Test Statistics |h3e|

Practically, we rarely explicitly pick a rejection region in terms of the
range of the data (:math:`\chi`).  It's usually much more convenient to pick a
rejection region in terms of a function of :math:`X` that produces a single number summarizing
the data called a **test statistic** (which we denote as :math:`T`).  Thus, our
expression for rejection region usually ends up looking something like this:

.. math::

    R = \big\{ x : T(x) > c \big\} \tag{6}

The value :math:`c` is called the **critical value** which determines
whether or not we retain or reject our null hypothesis.
So now the problem of hypothesis testing comes down to picking an appropriate 
test statistic :math:`T` and an appropriate value :math:`c` to minimize
our error rates (:math:`\alpha` and :math:`\beta`).

As mentioned above, minimizing :math:`\alpha` and :math:`\beta` are usually in
conflict, so what happens is we fix the level for :math:`\alpha` (usually values like
:math:`0.05` or :math:`0.01`), and find an appropriate :math:`T` and :math:`c`
so that :math:`\beta` is minimized (alternatively power is maximized).
Computing (and proving) that a test statistic has the highest power for a given 
an :math:`alpha` is quite complex so I won't mention much more of it here.
Most of the time though you won't have to actually come up with :math:`T` yourself
since many common situations have already been worked out.  The usual
procedure usually ends up being something along the lines of:

0. Define your null hypothesis (and the appropriate statistical model of your data).
1. Pick an appropriate :math:`\alpha`, e.g. :math:`0.05`.
2. Look up and compute the appropriate test statistic for your hypothesis/model e.g. `Z statistic <https://en.wikipedia.org/wiki/Z-test>`_.
3. Look up (or compute) the critical value :math:`c` based on :math:`\alpha` e.g. :math:`Z > 1.96`.
4. Retain/reject the null hypothesis based on the computed test statistic and critical value.

|h3| p-values and such |h3e|

Of course, just giving a retain/reject null hypothesis type answer isn't
very informative.  Instead, we might want to give the smallest :math:`\alpha`
that rejects the null hypothesis:

.. math::

    \text{p-value} = min\big\{ \alpha : T(X) \in R_{\alpha} \big\} \tag{7}

A p-value is basically a measure of evidence against :math:`H_0`.  
The smaller the p-value, the more evidence we have that :math:`H_0` is false.
Researchers usually use this scale for p-values:

|center|

================= =============================
p-value           Evidence  
================= =============================
:math:`<0.01`     very strong evidence against :math:`H_0`
----------------- -----------------------------
:math:`0.01-0.05` strong evidence against :math:`H_0`
----------------- -----------------------------
:math:`0.05-0.10` weak evidence against :math:`H_0`
----------------- -----------------------------
:math:`>0.10`     little or no evidence against :math:`H_0`
================= =============================

|centere|

Two important misconceptions about p-values:

* Nowhere in the above table do we say we have evidence for :math:`H_0`.
  *A p-value says nothing about evidence in favour of* :math:`H_0`.  A large
  p-value could mean that :math:`H_0` is true, or our test didn't have enough
  power.
* A p-value is not the probability that the null hypothesis is true (e.g. :math:`\text{p-value} \neq P(H_0 | Data)`).

A common way of stating what a p-value is (taken from *All of Statistics*):

    The p-value is the probability (under :math:`H_0`) of observing a value of
    the test statistic the same as or more extreme than what was actually
    observed.

Admittedly, this does not is not exactly line up with how we have looked at
:math:`\alpha` in terms of rejection regions, however, rest assured the
definitions do match up if you went through the derivations of the test
statistic and critical values.  Personally, I don't find the above definition
all that helpful because most people will conflate it with :math:`P(H_0|data)` 
just because both mention the word "probability".  

The way I like to think of it is simply a measure of evidence against :math:`H_0`
(but **not** for :math:`H_0`) according to the table above with no mention of probability.
In this way, we can remember the point of hypothesis testing is primarily a
procedure to help us prove our default or null hypothesis false.  Thinking this
way helps to remember that the null hypothesis is our default stance and the
test's aim is prove it false.

|h2| Conclusion |h2e|

While writing this post, I had to dig through the probabilistic foundations for
these techniques and it can get really deep!  I just scratched the surface,
enough to satisfy my intellectual curiosity and intuition (for now).
Hopefully, this post (and some of the references below) will help you along the
way too.


|h2| References and Further Reading |h2e|

* `All of Statistics: A Concise Course in Statistical Inference <http://link.springer.com/book/10.1007%2F978-0-387-21736-9>`_ by Larry Wasserman. (available free online)
* `Hypothesis Testing <http://www.stat.columbia.edu/~liam/teaching/4107-fall05/notes4.pdf>`_, Paninski, Intro. Math. Stats., December 6, 2005.
* Wikipedia: `Statistical Model <https://en.wikipedia.org/wiki/Statistical_model>`_, `Statistical Inference <https://en.wikipedia.org/wiki/Statistical_inference>`_, `Non parametric Statistics <https://en.wikipedia.org/wiki/Nonparametric_statistics>`_, `Statistical Hypothesis Testing <https://en.wikipedia.org/wiki/Statistical_hypothesis_testing>`_, `Statistical Power <https://en.wikipedia.org/wiki/Statistical_power>`_, `Sufficient Statistic <https://en.wikipedia.org/wiki/Sufficient_statistic>`_, `Null Hypothesis <https://en.wikipedia.org/wiki/Null_hypothesis>`_.

|br|

.. [1] Taking note that no model can truly represent reality leading to the aphorism: `All models are wrong <https://en.wikipedia.org/wiki/All_models_are_wrong>`_.

.. [2] `Inferential statistics <https://en.wikipedia.org/wiki/Statistical_inference>`_ is in contrast to `descriptive statistics <https://en.wikipedia.org/wiki/Descriptive_statistics>`_, which only tries to describe the sample or observations -- not estimate a probability distribution.  Examples of this are measures of central tendency (like mean or median), or measures of variability (such as standard deviation or min/max values).  Note that although the mean of a sample is a descriptive statistic, it is also an estimate for the expected value of a given distribution, thus used in statistical inference.  Similarly for the other descriptive statistics.

.. [3] There is a great chart in *All of Statistics* that shows the difference between statistics and computer science/data mining terminology on page xi of the preface.  It's very illuminating to contrast the two especially since terms like estimation, learning, covariates, hypothesis are thrown around very casually in their respective literature.  I come more from a computer science/data mining and learned most of my stats afterwards so it's great to see all these terms with their definitions in one place.

.. [4] Might be obvious but let's state it explicitly: *distribution* refers to the cumulative distribution function (CDF), and *density* refers to the probability density function (PDF).

.. [5] In fact, most of the time :math:`\mathfrak{F}` will not contain :math:`F` since as we mentioned above, the "true" distribution is probably much more complex than any model we could come up with.

.. [6] This categorization is given in *All of Statistics*, Section 6.3: Fundamental Concepts in Inference.  I've found it quite a good way to think about statistics from a high level.

.. [7] An important note outlined in *All of Statistics* about :math:`\theta`, point estimators and confidence intervals is that :math:`\theta` is fixed.  Recall, that our data is drawn from a "true" distribution that has (theoretically) *exact* parameters.  So there is a single fixed, albeit unknown, value of :math:`\theta`.  The randomness comes in through our observations.  Each observation, :math:`X_i`, is drawn (randomly) from the "true" distribution so by definition a random variable.  This means our point estimators :math:`\widehat{\theta}_n` and confidence intervals :math:`C_n` are also random variables since they are functions of random variables. |br| |br| This can all be a little confusing, so here's another way to think about it:  Say we have a "true" distribution, and we're going to draw :math:`n` samples from it.  Ahead of time, we don't know what the values of those observations are going to be but we know they will follow the "true" distribution.  Thus, the :math:`n` samples are :math:`n` random variables, each distributed according to the "true" distribution.  We can then take those :math:`n` variables and combine them into a function (e.g. a point estimator like a mean) to get a estimator.  This estimator, before we know the actual values of the :math:`n` variables, will also be a random variable.  However, what usually happens is that the values of the :math:`n` samples are actually observed, so we plugs these realizations into our point *estimator* (i.e. the function of the :math:`n` observations) to get a point *estimate* -- a deterministic value.  One reason we make this distinction is so that we can compute properties of our point estimator like bias and variance.  So long story short, the point estimator is a random variable where after having realized values of the observations, we can use it to get a single fixed number called a point estimate.

.. [8] Interestingly, it's very difficult to prove something to be true, whereas much easier to prove it false.  The reason is that many useful statements we want to prove are universally quantified (think of statements that use the word "all").  An example made famous by Nassim Nicholas Taleb is the "black swan" problem.  It's almost impossible to prove the statement "all swans are white" because you'd literally have to check the colour every single swan.  However, it's quite easy to prove it false by finding a single counter-example: a single black swan.  That's why the scientific method and hypothesis testing is such a good framework.  Knowing that it's difficult to prove things universally true, it sets itself up to weed out poor models of reality by allowing a systematic way of finding counter-examples (at least that's one way of looking at it).

.. [9] It's probably fair that when learning elementary hypothesis testing that you don't learn about the probabilistic interpretation.  For most students, they will never have to use hypothesis testing beyond rote application of standard tests.  However from an understanding perspective, I find this rather unappealing.  I at least like to have an intuition about how a method works rather than just a mechanical process thus this blog post.

.. [10] Think about a procedure that always rejects the null hypothesis i.e. a rejection consisting of the entire space.  In this case, our :math:`\alpha = 1` but :math:`\beta=0` because we are always correctly rejecting the null hypothesis when it is false.  Similarly if :math:`\beta = 1`.  Of course, this choice of rejection region is absolutely useless so we want to pick something a bit smarter.



