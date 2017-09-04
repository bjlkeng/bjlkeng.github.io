.. title: Semi-supervised Learning with Variational Autoencoders
.. slug: semi-supervised-learning-with-variational-autoencoders
.. date: 2017-08-22 08:45:47 UTC-04:00
.. tags: variational calculus, autoencoders, Kullback-Leibler, generative models, semi-supervised learning, inception, PCA, CNN, CIFAR10, mathjax
.. category: 
.. link: 
.. description: A post on semi-supervised learning with variational autoencoders.
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


In this post, I'll be continuing on this variational autoencoder (VAE) line of
exploration
(previous posts: `here <link://slug/variational-autoencoders>`__ and
`here <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__) by
writing about how to use variational autoencoders to do semi-supervised
learning.  In particular, I'll be explaining the technique used in
"Semi-supervised Learning with Deep Generative Models" by Kingma et al.
I'll be digging into math (hopefully being more explicit than the paper),
giving a bit more background on the variational lower bound as well as
my usual attempt at giving some more intuition.
I've also put some notebooks on Github that compare the VAE methods
with others such as PCA, CNNs, and pre-trained models.  Enjoy!

.. TEASER_END

|h2| Semi-supervised Learning |h2e|

`Semi-supervised learning <https://en.wikipedia.org/wiki/Semi-supervised_learning>`__
is a set of techniques used to make use of unlabelled data in supervised learning
problems (e.g. classification and regression).  Semi-supervised learning
falls in between unsupervised and supervised learning because you make use
of both labelled and unlabelled data points.

If you think about the plethora of data out there, most of it is unlabelled.
Rarely do you have something in a nice benchmark format that tells you exactly
what you need to do.  As an example, there are billions (trillions?) of
unlabelled images all over the internet but only a tiny fraction actually have
any sort of label.  So our goals here is to get the best performance with a 
tiny amount of labelled data.

Humans somehow are very good at this.  Even for those of us who haven't seen
one, I can probably show you a handful of 
`ant eater <http://blog.londolozi.com/2012/08/31/the-pantanal-series-walking-with-a-giant-anteater/>`__
images and you can probably classify them pretty accurately.  We're so good at
this because our brains have learned common features about what we see that
allow us to quickly categorize things into buckets like ant eaters.  For machines 
it's no different, somehow we want to allow a machine to learn some additional
(useful) features in an unsupervised way to help the actual task of which we have
very few examples.

|h2| Variational Lower Bound |h2e|

In my post on `variational Bayesian methods <http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/>`__, 
I discussed how to derive how to derive the variational lower bound but I just
want to spend a bit more time on it here to explain it on a bit more of a high
level.  In a lot of ML papers, they take for granted the "maximization of the
variational lower bound", so I just want to give a bit of intuition behind it.

Let's start off with the high level problem.  Recall, we have some data
:math:`X`, a generative probability model :math:`P(X|\theta)` that shows us how
to randomly sample (e.g. generate) data points that follow the distribution of
:math:`X`, assuming we know the "magic values" of the :math:`\theta`
parameters.  We can see Bayes theorem in Equation 1 (small :math:`p` for
densities):

.. math::

   p(\theta|X) &= \frac{p(X|\theta)p(\theta)}{p(X)} \\
               &= \frac{p(X|\theta)p(\theta)}{\int_{-\infty}^{\infty} p(X|\theta)p(\theta) d\theta} \\
               &= \frac{\text{likelihood}\cdot\text{prior}}{\text{evidence}} \\
               \tag{1}

Our goal is to find the posterior, :math:`P(\theta|X)`, that tells us the
distribution of the :math:`\theta` parameters, which sometimes is the end
goal (e.g. the cluster centers and mixture weights for a gaussian mixture models),
or we might just want the parameters so we can use :math:`P(X|\theta)` to generate
some new data points (e.g. use variational autoencoders to generate a new image).
Unfortunately, this problem is intractable (mostly the denominator) -- for all
but the simplest problems, we can't solve it nicely using an analytical
solution.  

Our solution?  Approximation! We'll approximate :math:`P(\theta|X)` by another
function :math:`Q(\theta|X)` (it's usually conditioned on :math:`X` but not
necessarily).  And (relatively) fast because we can assume a particular shape
for :math:`Q(\theta|X)` and turn the inference problem into an optimization
problem.  Of course, it can't be just a random function, we want it to be as
close as possible to :math:`P(\theta|X)` as possible, which will depend on the
structural form of :math:`Q(\theta|X)` (how much flexibility it has) as well as
our technique to find it and our metric of "closeness".

In terms of "closeness", the standard way of measuring it is to use
`KL divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__,
which we can neatly write down here:

.. math::

  D_{KL}(Q||P) &= \int_{-\infty}^{\infty} q(\theta|X) \log\frac{q(\theta|X)}{p(\theta|X)} d\theta \\
               &= \int_{-\infty}^{\infty} q(\theta|X) \log\frac{q(\theta|X)}{p(\theta,X)} d\theta + 
                  \int_{-\infty}^{\infty} q(\theta|X) \log{p(X)} d\theta \\
               &= \int_{-\infty}^{\infty} q(\theta|X) \log\frac{q(\theta|X)}{p(\theta,X)} d\theta + 
                  \log{p(X)} \\
              \tag{2}

Rearranging, dropping the KL divergence term and putting it in terms of an
expectation of :math:`q(\theta)`, we get what's called the *Evidence Lower
Bound* (ELBO) for a single data point :math:`X`:

.. math::

  \log{p(X)} &\geq -E_q\big[\log\frac{q(\theta|X)}{p(\theta,X)}\big]  \\
             &= E_q\big[\log p(\theta,X) - \log q(\theta|X)\big] \\
             &= E_q\big[\log p(X|\theta) + \log p(\theta) - \log q(\theta|X)\big] \\
             &= E_q\big[\text{likelihood} + \text{prior} - \text{approx. posterior} \big] \\
              \tag{3}

If you have multiple data points, you can just sum over them because we're 
in :math:`\log` space (assuming independence between data points).

In a lot of papers, you'll see that people will go straight to talking about
optimizing the ELBO whenever talking about variational inference.  And if you
look at it in isolation, you can gain some intuition of how it works:

* It's a lower bound on the evidence, that is, it's a lower bound on the
  probability of your data occurring given your model.
* Maximizing the ELBO is equivalent to minimizing the KL divergence.
* The first two terms try to maximize the MAP estimate (likelihood + prior).
* The last term tries to ensure :math:`Q` is diffuse (`maximize information entropy <link://slugmaximum-entropy-distributions>`__).

There's a pretty good presentation on this from NIPS 2016 by Blei et. al which
I've linked below if you want more details.  You can also check out my previous
post on `variational inference <link://variational-bayes-and-the-mean-field-approximation>`__,
if you want some more nitty-gritty details of how to derive everything
(although I don't put it in ELBO terms).


|h2| A Vanilla VAE for Semi-Supervised Learning (M1 Model) |h2e|

I won't go over all the details of variational autoencoders again, you
can check out my previous post for that 
(`variational autoencoders <link://slug/variational-autoencoders>`__).
The high level idea is pretty easy to understand though.  A variational
autoencoder defines a generative model for your data which basically says take
an isotropic standard normal distribution (:math:`Z`), we run it through a deep
net (defined by :math:`g`) to produce the observed data (:math:`X`).  The hard
part is figuring out how to train it.

Using the autoencoder analogy, the generative model is the "decoder" since
you're starting from a latent state and translating it to the observed data.  A
VAE also has an "encoder" part that is used to help train the decoder.  It goes
from observed values to a latent state (:math:`z` to :math:`X`).  A keen
observer will notice that this is actually our variational approximation of the
posterior (:math:`z|X`), which coincidentally is also a neural network (defined
by :math:`g_{z|X}`).  This is visualized in Figure 1.

TODO: Use another figure for 1

After our VAE has been fully trained, it's easy to see how we can just use the
"encoder" to directly help with semi-supervised learning:

1. Transform our observed data (:math:`X`) into the latent space defined by the
   :math:`Z` variables using *all* our data points (labelled and unlabelled).
2. Solve a standard supervised learning problem on the *labeled* data using 
   :math:`(Y, Z)` pairs (where :math:`Y` is our label).

Intuitively, the latent space defined by :math:`z` should capture some useful
information about our data such that it's easily separable in our supervised
learning problem.  This technique is defined as M1 model in the Kingma paper.
As you may have noticed though, step 1 doesn't directly involve any of the
:math:`y` labels; the two steps are disjoint.  Kingma also introduces another
model "M2" that attempts to solve this problem.


|h2| Extending the VAE for Semi-Supervised Learning (M2 Model) |h2e|

In the M1 model, we basically ignored our labelled data in our VAE.
The M2 model (from the Kingma paper) explicitly takes it into account.  Let's
take a look at the generative model (i.e. the "decoder"):

.. math::

    p({\bf x}|y,{\bf z}) = f({\bf x}; y, {\bf z}, \theta) \\
    p({\bf z}) = \mathcal{N}({\bf z}|0, I) \\
    p(y|{\bf \pi}) = \text{Cat}(y|{\bf {\bf \pi}}) \\
    p({\bf \pi}) = \text{SymDir}(\alpha) \\
    \tag{4}

where:

* :math:`{\bf x}` is a vector of normally distributed observed variables
* :math:`f({\bf x}; y, {\bf z}, \theta)` is a suitable likelihood function such
  as a Gaussian or Bernoulli.  We use a deep net to approximate it based on
  inputs :math:`{\bf z}, y` with network weights defined by :math:`\theta`.
* :math:`{\bf z}` is a vector latent variables (same as vanilla VAE)
* :math:`y` is one-hot encoded categorical variable representing our class
  labels, whose relative probabilities are parameterized by :math:`{\bf \pi}`.
* :math:`\text{SimDir}` is `Symmetric Dirichlet <https://en.wikipedia.org/wiki/Dirichlet_distribution#Special_cases>`__ distribution with hyper-parameter :math:`\alpha` (a conjugate prior for categorical/multinomial variables)

How do we do use this for semi-supervised learning you ask?  The basic gist of
it is, we will define a approximate posterior function 
:math:`q_\phi(y|{\bf x})` using a deep net that is basically a classifier.
However the genius is that we can train this classifier for both labelled *and*
unlabelled data by just training this extended VAE.  Figure 2 shows a
visualization of the network.

TODO: Make another figure for semi-supervised autoencoder Figure 2

Now the interesting part is that we have two cases: one where we observe the
:math:`y` labels and one where we don't.  We have to deal with them differently
when constructing the approximate posterior :math:`q` as well as in the
variational objective. 

|h3| Variational Objective with Unlabelled Data |h3e|

For any variational inference problem, we need to start with our approximate
posterior.  In this case, we'll treat :math:`{\bf z}, y` as the unknown
latent variables, and perform variational inference (i.e. define approximate
posteriors) over them.  Notice that we excluded :math:`\pi` because we don't
really care what its posterior is in this case.

We'll assume the approximate posterior :math:`q_{\phi}({\bf z}, y|{\bf x})` has
a fully factorized form as such:

.. math::

    q_{\phi}({\bf z}, y|{\bf x}) &= q_{\phi}({\bf z}|{\bf x})q_{\phi}(y|{\bf x}) \\
    q_{\phi}({\bf z}|{\bf x}) &= 
        \mathcal{N}({\bf z}| {\bf \mu}_{\phi}({\bf x}),
                             diag({\bf \sigma}^2_{\phi}({\bf x}))) \\
    q_{\phi}(y|{\bf x}) &= \text{Cat}(y|\pi_{\phi}({\bf x})) \\
    \tag{5}

where 
:math:`{\bf \mu}_{\phi}({\bf x}), {\bf \sigma}^2_{\phi}({\bf x}), \pi_{\phi}({\bf X}`)
all define neural networks parameterized by :math:`\phi` that we will learn
(in the actual implementation, 
:math:`{\bf \mu}_{\phi}({\bf x}), {\bf \sigma}^2_{\phi}({\bf x})` and 
:math:`\pi_{\phi}({\bf X})` are actually different networks).
Here, :math:`\pi_{\phi}({\bf X})` should not be confused with our actual
parameter :math:`{\bf \pi}` above, the former is an estimate coming out of our
network, the latter is our prior distribution as a symmetric Dirichlet.

From here, we use the ELBO to determine our variational objective for a single
data point:

.. math::

    \log p_{\theta}({\bf x}) &\geq E_{q_\phi(y, {\bf z}|{\bf x})}\bigg[ 
        \log p_{\theta}({\bf x}|y, {\bf z}) + \log p_{\theta}(y)
          + \log p_{\theta}({\bf z}) - \log q_\phi(y, {\bf z}|{\bf x})
    \bigg] \\
    &= E_{q_\phi(y|{\bf x})}\bigg[
       E_{q_\phi({\bf z}|{\bf x})}\big[ 
        \log p_{\theta}({\bf x}|y, {\bf z}) + K_1
        + \log p_{\theta}({\bf z})  - \log q_\phi(y|{\bf x}) - \log q_\phi({\bf z}|{\bf x})
       \big]
    \bigg] \\
    &= E_{q_\phi(y|{\bf x})}\bigg[
       E_{q_\phi({\bf z}|{\bf x})}\big[ 
        \log p_{\theta}({\bf x}|y, {\bf z}) 
       \big]
        + K_1
        - KL[q_{\phi}({\bf z}|{\bf x})||p_{\theta}({\bf z})]
        - \log q_{\phi}(y|{\bf x})
    \bigg] \\
    &= E_{q_\phi(y|{\bf x})}\big[ -\mathcal{L({\bf x}, y)} 
        - \log q_{\phi}(y|{\bf x})
        \big] \\   
    &= \sum_y q_\phi(y|{\bf x})(-\mathcal{L}({\bf x}, y)) 
        + q_\phi(y|{\bf x}) \log q_\phi(y|{\bf x}) \\
    &= \sum_y q_\phi(y|{\bf x})(-\mathcal{L}({\bf x}, y)) 
        + \mathcal{H}(q_\phi(y|{\bf x})) \tag{6}

Going through line by line, we factor our :math:`q_\phi` function
into the separate :math:`y` and :math:`{\bf z}` parts for both the expectation
and the :math:`\log`. Notice we also absorb :math:`\log p_\theta(y)` into a
constant because :math:`p(y) = p(y|{\bf \pi})p(\pi)`, a 
`Dirichlet-multinomial <https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution#Specification>`__
distribution, and simplifies to a constant (alternatively, our model's assumption is that :math:`y`'s are equally likely to happen).

Next, we group a bunch of terms together, noticing that some form a KL
distribution between :math:`q_{\phi}({\bf z}|{\bf x})` and
:math:`p_{\theta}({\bf z})`, and call the group 
:math:`\mathcal{L}({\bf x}, y)`.  This latter term is essentially the same
variational objective we used for a vanilla variational autoencoder (sans the 
reference to :math:`y`).  Finally, we explicitly write out the expectation
with respect to :math:`y`.  I won't write out all the details for how
to compute it, for that you can look at my previous post for 
:math:`\mathcal{L}({\bf x}, y)`, and the implementation notebooks for the rest.
The loss functions are pretty clearly labelled so it shouldn't be too hard to
map it back to these equations.

So Equation 6 defines our objective function for our VAE, which will
simultaneously train both the :math:`\theta` parameters of the "decoder" network
as well as the approximate posterior "encoder" :math:`\phi` parameters relating
to :math:`y, {\bf z}`.

|h3| Variational Objective with Labelled Data |h3e|

So here's where it gets a bit trickier because this part was glossed over in
the paper.  In particular, when training with labelled data, you want to make
sure you train both the :math:`y` *and* the :math:`{\bf z}` networks at the
same time.  It's actually easy to leave out the :math:`y` network since you
the observations for :math:`y` are not needed to generate :math:`{\bf x}` on the 
output, essentially you can just use :math:`\mathcal{L}(x,y)` as your loss
function (it's very similar to a vanilla VAE except with extra input variables
defined by :math:`y`).

Now of course the *whole* point of semi-supervised learning is to learn a
mapping using labelled data from :math:`{\bf x}` to :math:`y` so it's pretty
silly not to train that part of your VAE (i.e. the classifier part).  So
Kingma et al. add an extra loss term initially describing it as a fix to this
problem.  Then, they add an innocent throw-away line that this actually can be
derived by performing variational inference over :math:`\pi`.  Of course, it's
actually true but it's not that straightforward to derive!  Well, I worked out
all the gory the details, so here's my presentation of deriving the variational
objective with labelled data.

-----

For the case when we have both :math:`(x,y)` points, we'll treat both :math:`z`
and :math:`{\bf \pi}` as unknown latent variables and perform variational
inference for both :math:`\bf{z}` and :math:`{\bf \pi}` using a fully
factorized posterior dependent *only* on :math:`{\bf x}`.

.. math::

    q({\bf z}, {\bf \pi}) &= q({\bf z}, {\bf \pi}|{\bf x}) \\
              &= q({\bf z}|X) * q({\bf \pi}|{\bf x}) \\
    q({\bf z}|{\bf x})  &= N({\bf \mu}_{\phi}({\bf x}), {\bf \sigma}^2_{\phi}({\bf x})) \\
    q({\bf \pi}|{\bf x})  &= SymDir(\alpha + {\bf \pi}_{\phi}({\bf x})) 
    \tag{7}
    
Remember we can define our approximate posteriors however we want, so we
explicitly choose to have :math:`{\bf \pi}` to depend *only* on :math:`{\bf x}`
and *not* on our observed :math:`y`.  Why you ask?  It's because we want to make
sure our :math:`\phi` parameters of our classifier are trained when we have
labelled data.

The last line of defining :math:`q({\bf \pi}|{\bf x})` as as symmetric Dirichlet
comes from the fact that the 
`Dirichlet is the conjugate prior <https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical.2Fmultinomial>`__ 
of a multinomial distribution.  So we treat :math:`{\bf \pi}_{\phi}({\bf x})`
as a single "observed trial" of a multinomial distribution, which leads to 
the distribution you see above.

As before, we start with the ELBO to determine our variational objective for a
single data point :math:`({\bf x},y)`:

.. math::

    \log p_{\theta}({\bf x}, y) &\geq 
        E_{q_\phi({\bf \pi}, {\bf z}|{\bf x}, y)}\bigg[ 
        \log p_{\theta}({\bf x}|y, {\bf z}) 
        + \log p_{\theta}({\bf \pi}|y)
        + \log p_{\theta}(y)
        + \log p_{\theta}({\bf z}) 
        - \log q_\phi({\bf \pi}, {\bf z}|{\bf x}, y)
    \bigg] \\
    &= E_{q_\phi({\bf z}|{\bf x})}\bigg[ 
        \log p_{\theta}({\bf x}|y, {\bf z})
        + \log p_{\theta}(y)
        + \log p_{\theta}({\bf z}) 
        - \log q_\phi({\bf z}|{\bf x})
    \bigg] \\
    &\quad + E_{q_\phi({\bf \pi}|{\bf x})}\bigg[ 
        \log p_{\theta}({\bf \pi}|y)
        - \log q_\phi({\bf \pi}|{\bf x})
    \bigg] \\
    &= -\mathcal{L}(x,y)
       - KL[q_\phi({\bf \pi}|{\bf x})||p_{\theta}({\bf \pi}|y)] \\
    &= -\mathcal{L}(x,y) 
  + \mathcal{H}(q_\phi(y|{\bf x}))   
  + \alpha \log q_\phi(y|{\bf x}) + K_2
    \tag{8}

Going line by line, we off with the ELBO, expanding all the priors.  The one
trick we do is instead of expanding the joint distribution of 
:math:`y,{\bf \pi}` conditioned on :math:`\pi` (i.e.
:math:`p_{\theta}(y, {\bf \pi}) = p_{\theta}(y|{\bf \pi})p_{\theta}({\bf \pi})`),
we instead expand using the posterior: :math:`\log p_{\theta}({\bf \pi}|y)`.
The posterior in this case is again a
`Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical.2Fmultinomial>`__
because it's the conjugate prior of :math:`y`'s categorical/multinomial distribution.

Next, we just rearrange and factor :math:`q_\phi`, both in the :math:`\log`
term as well as the expectation.  We notice that the first part is exactly our
:math:`\mathcal{L}` loss function from above and the rest is a KL divergence
between our :math:`\pi` posterior and our approximate posterior.  The
last simplification of the KL divergence is a bit verbose so I've put it in
Appendix A.


* Show how to dervied additional loss term using Dirichlet prior (maybe appendix)?
* They also have a "M1" + "M2" model that I didn't try

|h2| Semi-supervised Results |h2e|

* Talk about comparison methods: PCA + SVM, CNN, Inception + additional layers,
  M1, M2
* Mention some other common techniques (didn't try): data augmentation, others?
* Used CIFAR10
* Show comparison table
* Talk about implementation a bit, point to notebooks

|h2| Conclusion |h2e|

TODO

|h2| Further Reading |h2e|

* Previous Posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, `Variational Bayes and The Mean-Field Approximation <link://variational-bayes-and-the-mean-field-approximation>`__, `Maximum Entropy Distributions <link://slugmaximum-entropy-distributions>`__
* Wikipedia: `Semi-supervised learning <https://en.wikipedia.org/wiki/Semi-supervised_learning>`__, `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__
* "Variational Inference: Foundations and Modern Methods", Blei, Ranganath, Mohamed, `NIPS 2016 Tutorial <https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf>`__.
* "Semi-supervised Learning with Deep Generative Models", Kingma, Rezende, Mohamed, Welling, https://arxiv.org/abs/1406.5298
* Github report for "Semi-supervised Learning with Deep Generative Models", https://github.com/dpkingma/nips14-ssl/
  

|h2| Appendix A: KL Divergence of
:math:`q_\phi({\bf \pi}|{\bf x})||p_{\theta}({\bf \pi}|y)` 
|h2e|

Notice that the two distributions in question are both
`Dirichlet distributions <https://en.wikipedia.org/wiki/Dirichlet_distribution>`__:

.. math::

    q_{\phi}({\bf \pi}|{\bf x})  &= Dir(\alpha_q {\bf \pi}_{\phi}({\bf x})) \\
    p_{\theta}({\bf \pi}|y)  &= Dir(\alpha + {\bf c}_y) \\
    \tag{A.1}

where :math:`{\bf c}_y` is a vector with 0's and a single 1 representing the
categorical observation we have.
In fact, both distributions are just the conjugate prior of a single
observation of a categorical variable; :math:`y` in one case and 
:math:`{\bf \pi}_{\phi}({\bf x})` in another.  

Let's take a look at the formula for 
`KL divergence between two Symmetric Dirichlets <http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/>`__
parameterized by vectors :math:`{\bf \alpha}` and :math:`{\bf \beta}`:

.. math::

    KL(p||q) =  \log \Gamma(\alpha_0) - \sum_{k=1}^{K} \log \Gamma(\alpha_k)
             - \log \Gamma(\beta_0) + \sum_{k=1}^{K} \log \Gamma(\beta_k)
             + \sum_{k=1}^K (\alpha_k - \beta_k)E_{p(x)}[\log x_k] \\
    \tag{A.2}

where :math:`\alpha_0=\sum_k \alpha_k` and :math:`\beta_0=\sum_k \beta_k`.

Our problem simplifies a lot because we have the constant :math:`alpha` in
common for both sides which makes it a mostly symmetric problem.  Substituting
Equation A.1 into A.2, we have:

.. math::

    KL[q_\phi({\bf \pi}|{\bf x})||p_{\theta}({\bf \pi}|y)] 
            &= \log \Gamma(\alpha_q) 
            - \sum_{k=1}^{K} \log \Gamma(\alpha_q \pi_{\phi,k}({\bf x})) \\
            &\quad - \log \Gamma(K\alpha + 1)
                + \sum_{k=1}^{K} \log \Gamma(\alpha + c_{y,k}) \\
            &\quad + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - \alpha - {\bf c}_{y,k})
              E_{q_\phi({\bf \pi}|{\bf x})}[\log \pi_k] \\
    &\leq K_2 + \sum_{k=1}^K (\pi_{\phi,k}({\bf x}) - {\bf c}_{y,k})
       E_{q_\phi({\bf \pi}|{\bf x})}[\log \pi_k] \\
    &= K_2 
       + E_{q_\phi({\bf \pi}|{\bf x})}[
        \sum_{k=1}^K \pi_{\phi,k}({\bf x})\log \pi_k
        - \sum_{k=1}^K {\bf c}_{y,k}\log \pi_k] \\
    &\approx K_2
       + \sum_{k=1}^K \pi_{\phi,k}({\bf x})\log \pi_{\phi,k}({\bf x})
        - \sum_{k=1}^K {\bf c}_{y,k}\log \pi_{\phi,k}({\bf x}) \\
    &= K_2
       + \sum_{k=1}^K \pi_{\phi,k}({\bf x})\log \pi_{\phi,k}({\bf x})
        - \sum_{k=1}^K \log \pi_{\phi,k}({\bf x})^{{\bf c}_{y,k}} \\
    &= K_2
       + \mathcal{H}(q_\phi(y|{\bf x})) - \log q_\phi(y|{\bf x}) \\
    \tag{A.3}

Going line by line, we first fill in all our pseudo-count parameters
for the Dirichlet.  We can see the first four terms are *almost* constants
since both :math:`{\bf \pi_\phi}` and :math:`{\bf c}_y` are all upper bounded
by :math:`1`.  So in the next line, we just absorb them into a constant
:math:`K_2`.  Next, we rearrange and use the same trick we used for vanilla
`variational autoencoders <link://slug/variational-autoencoders>`__:
approximate the expectation the mean, which is :math:`\pi_{\phi,k}({\bf x})` [1]_.

I mentioned above that 
:math:`KL[q_\phi({\bf \pi}|{\bf x})||p_{\theta}({\bf \pi}|y)]` 
can be simplified to :math:`\log q_\phi(y|{\bf x})`


|br|

.. [1] This is I'm actually a bit unsure about.  If we're using the exact same sampling trick, we should actually be sampling from a Dirichlet
