.. title: Semi-supervised Learning with Variational Autoencoders
.. slug: semi-supervised-learning-with-variational-autoencoders
.. date: 2017-09-11 08:40:47 UTC-04:00
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
I'll be digging into the math (hopefully being more explicit than the paper),
giving a bit more background on the variational lower bound, as well as
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
want to spend a bit more time on it here to explain it in a different way.  In
a lot of ML papers, they take for granted the "maximization of the variational
lower bound", so I just want to give a bit of intuition behind it.

Let's start off with the high level problem.  Recall, we have some data
:math:`X`, a generative probability model :math:`P(X|\theta)` that shows us how
to randomly sample (e.g. generate) data points that follow the distribution of
:math:`X`, assuming we know the "magic values" of the :math:`\theta`
parameters.  We can see Bayes theorem in Equation 1 (small :math:`p` for
densities):

.. math::

   p(\theta|X) &= \frac{p(X|\theta)p(\theta)}{p(X)} \\
               &= \frac{p(X|\theta)p(\theta)}{\int_{-\infty}^{\infty} p(X|\theta)p(\theta) d\theta} \\
   \text{posterior}   &= \frac{\text{likelihood}\cdot\text{prior}}{\text{evidence}} \\
               \tag{1}

Our goal is to find the posterior, :math:`P(\theta|X)`, that tells us the
distribution of the :math:`\theta` parameters, which sometimes is the end
goal (e.g. the cluster centers and mixture weights for a Gaussian mixture models),
or we might just want the parameters so we can use :math:`P(X|\theta)` to generate
some new data points (e.g. use variational autoencoders to generate a new image).
Unfortunately, this problem is intractable (mostly the denominator) for all
but the simplest problems, that is, we can't get a nice closed-form solution.  

Our solution?  Approximation! We'll approximate :math:`P(\theta|X)` by another
function :math:`Q(\theta|X)` (it's usually conditioned on :math:`X` but not
necessarily).  And solving for :math:`Q` is (relatively) fast because we can
assume a particular shape for :math:`Q(\theta|X)` and turn the inference
problem (i.e. finding :math:`P(\theta|X)`) into an optimization problem (i.e.
finding :math:`Q`).  Of course, it
can't be just a random function, we want it to be as close as possible to
:math:`P(\theta|X)`, which will depend on the structural form of
:math:`Q(\theta|X)` (how much flexibility it has), our technique to find it,
and our metric of "closeness".

In terms of "closeness", the standard way of measuring it is to use
`KL divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__,
which we can neatly write down here:

.. math::

  D_{KL}(Q||P) &= \int_{-\infty}^{\infty} q(\theta|X) \log\frac{q(\theta|X)}{p(\theta|X)} d\theta \\
               &= \int_{-\infty}^{\infty} q(\theta|X) \log\frac{q(\theta|X)}{p(\theta,X)} d\theta + 
                  \int_{-\infty}^{\infty} q(\theta|X) \log{p(X)} d\theta \\
               &= \int_{-\infty}^{\infty} q(\theta|X) \log\frac{q(\theta|X)}{p(\theta,X)} d\theta + 
                  \log{p(X)} \\
               &= E_q\big[\log\frac{q(\theta|X)}{p(\theta,X)}\big] + \log p(X) \\
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

In a lot of papers, you'll see that people will go straight to optimizing the
ELBO whenever they are talking about variational inference.  And if you look at
it in isolation, you can gain some intuition of how it works:

* It's a lower bound on the evidence, that is, it's a lower bound on the
  probability of your data occurring given your model.
* Maximizing the ELBO is equivalent to minimizing the KL divergence.
* The first two terms try to maximize the MAP estimate (likelihood + prior).
* The last term tries to ensure :math:`Q` is diffuse (`maximize information entropy <link://slug/maximum-entropy-distributions>`__).

There's a pretty good presentation on this from NIPS 2016 by Blei et al. which
I've linked below if you want more details.  You can also check out my previous
post on `variational inference <link://slug/variational-bayes-and-the-mean-field-approximation>`__,
if you want some more nitty-gritty details of how to derive everything
(although I don't put it in ELBO terms).


|h2| A Vanilla VAE for Semi-Supervised Learning (M1 Model) |h2e|

I won't go over all the details of variational autoencoders again, you
can check out my previous post for that 
(`variational autoencoders <link://slug/variational-autoencoders>`__).
The high level idea is pretty easy to understand though.  A variational
autoencoder defines a generative model for your data which basically says take
an isotropic standard normal distribution (:math:`Z`), run it through a deep
net (defined by :math:`g`) to produce the observed data (:math:`X`).  The hard
part is figuring out how to train it.

Using the autoencoder analogy, the generative model is the "decoder" since
you're starting from a latent state and translating it into the observed data.  A
VAE also has an "encoder" part that is used to help train the decoder.  It goes
from observed values to a latent state (:math:`X` to :math:`z`).  A keen
observer will notice that this is actually our variational approximation of the
posterior (:math:`q(z|X)`), which coincidentally is also a neural network (defined
by :math:`g_{z|X}`).  This is visualized in Figure 1.

.. figure:: /images/vanilla_vae.png
  :width: 550px
  :alt: Vanilla Variational Autoencoder
  :align: center

  Figure 1: Vanilla Variational Autoencoder


After our VAE has been fully trained, it's easy to see how we can just use the
"encoder" to directly help with semi-supervised learning:

1. Train a VAE using *all* our data points (labelled and unlabelled), and
   transform our observed data (:math:`X`) into the latent space defined by the
   :math:`Z` variables.
2. Solve a standard supervised learning problem on the *labelled* data using 
   :math:`(Z, Y)` pairs (where :math:`Y` is our label).

Intuitively, the latent space defined by :math:`z` should capture some useful
information about our data such that it's easily separable in our supervised
learning problem.  This technique is defined as M1 model in the Kingma paper.
As you may have noticed though, step 1 doesn't directly involve any of the
:math:`y` labels; the steps are disjoint.  Kingma also introduces another
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

* :math:`{\bf x}` is a vector of our observed variables
* :math:`f({\bf x}; y, {\bf z}, \theta)` is a suitable likelihood function to
  model our output such as a Gaussian or Bernoulli.  We use a deep net to
  approximate it based on inputs :math:`y, {\bf z}` with network weights
  defined by :math:`\theta`.
* :math:`{\bf z}` is a vector latent variables (same as vanilla VAE)
* :math:`y` is a one-hot encoded categorical variable representing our class
  labels, whose relative probabilities are parameterized by :math:`{\bf \pi}`.
* :math:`\text{SimDir}` is `Symmetric Dirichlet <https://en.wikipedia.org/wiki/Dirichlet_distribution#Special_cases>`__ distribution with hyper-parameter :math:`\alpha` (a conjugate prior for categorical/multinomial variables)

How do we do use this for semi-supervised learning you ask?  The basic gist of
it is: we will define a approximate posterior function 
:math:`q_\phi(y|{\bf x})` using a deep net that is basically a classifier.
However the genius is that we can train this classifier for both labelled *and*
unlabelled data by just training this extended VAE.  Figure 2 shows a
visualization of the network.

.. figure:: /images/m2_vae.png
  :width: 550px
  :alt: M2 Variational Autoencoder for Semi-Supervised Learning
  :align: center

  Figure 2: M2 Variational Autoencoder for Semi-Supervised Learning

Now the interesting part is that we have two cases: one where we observe the
:math:`y` labels and one where we don't.  We have to deal with them differently
when constructing the approximate posterior :math:`q` as well as in the
variational objective. 

|h3| Variational Objective with Unlabelled Data |h3e|

For any variational inference problem, we need to start with our approximate
posterior.  In this case, we'll treat :math:`y, {\bf z}` as the unknown
latent variables, and perform variational inference (i.e. define approximate
posteriors) over them.  Notice that we excluded :math:`\pi` because we don't
really care what its posterior is in this case.

We'll assume the approximate posterior :math:`q_{\phi}(y, {\bf z}|{\bf x})` has
a fully factorized form as such:

.. math::

    q_{\phi}(y, {\bf z}|{\bf x}) &= q_{\phi}({\bf z}|{\bf x})q_{\phi}(y|{\bf x}) \\
    q_{\phi}(y|{\bf x}) &= \text{Cat}(y|\pi_{\phi}({\bf x})) \\
    q_{\phi}({\bf z}|{\bf x}) &= 
        \mathcal{N}({\bf z}| {\bf \mu}_{\phi}({\bf x}),
                             diag({\bf \sigma}^2_{\phi}({\bf x}))) \\
    \tag{5}

where 
:math:`{\bf \mu}_{\phi}({\bf x}), {\bf \sigma}^2_{\phi}({\bf x}), \pi_{\phi}({\bf X}`)
are all defined by neural networks parameterized by :math:`\phi` that we will learn.
Here, :math:`\pi_{\phi}({\bf X})` should not be confused with our actual
parameter :math:`{\bf \pi}` above, the former is a point-estimate coming out of
our network, the latter is a random variable as a symmetric Dirichlet.

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
    &= \sum_y \big[ q_\phi(y|{\bf x})(-\mathcal{L}({\bf x}, y)) 
        - q_\phi(y|{\bf x}) \log q_\phi(y|{\bf x}) \big] \\
    &= \sum_y q_\phi(y|{\bf x})(-\mathcal{L}({\bf x}, y)) 
        + \mathcal{H}(q_\phi(y|{\bf x})) \\
          \tag{6}

Going through line by line, we factor our :math:`q_\phi` function
into the separate :math:`y` and :math:`{\bf z}` parts for both the expectation
and the :math:`\log`. Notice we also absorb :math:`\log p_\theta(y)` into a
constant because :math:`p(y) = p(y|{\bf \pi})p(\pi)`, a 
`Dirichlet-multinomial <https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution#Specification>`__
distribution, and simplifies to a constant (alternatively, our model's assumption is that :math:`y`'s are equally likely to happen).

Next, we notice that some terms form a KL distribution between :math:`q_{\phi}({\bf
z}|{\bf x})` and :math:`p_{\theta}({\bf z})`. Then, we group a few terms
together and name it :math:`\mathcal{L}({\bf x}, y)`.  This
latter term is essentially the same
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
have the observations for :math:`y`, allowing you to ignore the classifier
network.

Now of course the *whole* point of semi-supervised learning is to learn a
mapping using labelled data from :math:`{\bf x}` to :math:`y` so it's pretty
silly not to train that part of your VAE using labelled data.  So Kingma et al.
add an extra loss term initially describing it as a fix to this problem.  Then,
they add an innocent throw-away line that this actually can be derived by
performing variational inference over :math:`\pi`.  Of course, it's actually
true (I think) but it's not that straightforward to derive!  Well, I worked out
the details, so here's my presentation of deriving the variational objective
with labelled data.

-----

For the case when we have both :math:`(x,y)` points, we'll treat both :math:`z`
and :math:`{\bf \pi}` as unknown latent variables and perform variational
inference for both :math:`\bf{z}` and :math:`{\bf \pi}` using a fully
factorized posterior dependent *only* on :math:`{\bf x}`.

.. math::

    q({\bf z}, {\bf \pi}) &= q({\bf z}, {\bf \pi}|{\bf x}) \\
              &= q({\bf z}|X) * q({\bf \pi}|{\bf x}) \\
    q({\bf z}|{\bf x})  &= N({\bf \mu}_{\phi}({\bf x}), {\bf \sigma}^2_{\phi}({\bf x})) \\
    q({\bf \pi}|{\bf x})  &= Dir(\alpha_q{\bf \pi}_{\phi}({\bf x})) 
    \tag{7}
    
Remember we can define our approximate posteriors however we want, so we
explicitly choose to have :math:`{\bf \pi}` to depend *only* on :math:`{\bf x}`
and *not* on our observed :math:`y`.  Why you ask?  It's because we want to make
sure our :math:`\phi` parameters of our classifier are trained when we have
labelled data.

As before, we start with the ELBO to determine our variational objective for a
single data point :math:`({\bf x},y)`:

.. math::

    \log p_{\theta}({\bf x}, y) &\geq 
        E_{q_\phi({\bf \pi}, {\bf z}|{\bf x}, y)}\bigg[ 
        \log p_{\theta}({\bf x}|y, {\bf z}, {\bf \pi}) 
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
    &= -\mathcal{L}({\bf x},y)
       - KL[q_\phi({\bf \pi}|{\bf x})||p_{\theta}({\bf \pi}|y)] \\
    &\geq -\mathcal{L}({\bf x},y) + \alpha \log q_\phi(y|{\bf x}) + K_2
    \tag{8}

where :math:`\alpha` is a hyper-parameter that controls the relative weight of how
strongly you want to train the discriminative classified (:math:`q_\phi(y|{\bf
x})`).  In the paper, they set it to :math:`\alpha=0.1N`


Going line by line, we start off with the ELBO, expanding all the priors.  The one
trick we do is instead of expanding the joint distribution of 
:math:`y,{\bf \pi}` conditioned on :math:`\pi` 
(i.e.  :math:`p_{\theta}(y, {\bf \pi}) = p_{\theta}(y|{\bf \pi})p_{\theta}({\bf \pi})`),
we instead expand using the posterior: :math:`p_{\theta}({\bf \pi}|y)`.
The posterior in this case is again a
`Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical.2Fmultinomial>`__
because it's the conjugate prior of :math:`y`'s categorical/multinomial distribution.

Next, we just rearrange and factor :math:`q_\phi`, both in the :math:`\log`
term as well as the expectation.  We notice that the first part is exactly our
:math:`\mathcal{L}` loss function from above and the rest is a KL divergence
between our :math:`\pi` posterior and our approximate posterior.  
The last simplification of the KL divergence is a bit verbose (and hand-wavy)
so I've put it in Appendix A.


|h3| Training the M2 Model |h3e|

Using Equations 6 and 8, we can derive a loss function as such (remember it's
the negative of the ELBO above):

.. math::

    \mathcal{J} = 
    \sum_{{\bf x} \in \mathcal{D}_{unlabelled}} \big[
        \sum_y q_\phi(y|{\bf x})(\mathcal{L}({\bf x}, y)) - \mathcal{H}(q_\phi(y|{\bf x}))
    \big]
    + \sum_{({\bf x},y) \in \mathcal{D}_{labelled}} \big[
        \mathcal{L}({\bf x},y) - \alpha \log q_\phi(y|{\bf x}) 
    \big] \\
    \tag{9}

With this loss function, we just train the network as you would expect.
Simply grab a mini-batch, compute the needed values in the network
(i.e. :math:`q(y|{\bf x}), q(z|{\bf x}), p({\bf x}|y, z)`), compute the loss
function above using the appropriate summation depending on if you have
labelled or unlabelled data, and finally just take the gradients to update our
network parameters :math:`\theta, \phi`.  The network is remarkably similar
to a vanilla VAE with the addition of the posterior on :math:`y`, and the
additional terms to the loss function.  The tricky part is dealing with
the two types of data (labelled and unlabelled), which I explain in the
implementation notes below.


|h2| Implementation Notes |h2e|

The notebooks I used are `here
<https://github.com/bjlkeng/sandbox/tree/master/notebooks/vae-semi_supervised_learning>`__
on Github.  I made one notebook for each experiment, so it should be pretty
easy for you to look around.  I didn't add as many comments as some of my
previous notebooks but I think the code is relatively clean and straightforward
so I don't think you'll have much trouble understanding it.

|h3| Variational Autoencoder Implementations (M1 and M2) |h3e|

The architectures I used for the VAEs were as follows: 

* For :math:`q(y|{\bf x})`, I used the `CNN example <https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py>`__ from Keras,
  which has 3 conv layers, 2 max pool layers, a softmax layer, with dropout and ReLU activation.
* For :math:`q({\bf z}|{\bf x})`, I used 3 conv layers, and 2 fully connected
  layers with batch normalization, dropout and ReLU activation.
* For :math:`p({\bf x}|{\bf z})` and :math:`p({\bf x}|y, {\bf z})`, I used a
  fully connected layer, followed by 4 transposed conv layers (the first 3 with
  ReLU activation the last with sigmoid for the output).

The rest of the details should be pretty straight forward if you look at the
notebook.

The one complication that I had was how to implement the training of the M2
model because you need to treat :math:`y` simultaneously as an input and an
output depending if you have labelled or unlabelled data.  I still wanted to
use Keras and didn't want to go as low level as TensorFlow, so I came up with
a workaround: train two networks (with shared layers)!

So basically, I have one network for labelled data and one for unlabelled data.
They both share all the same components (:math:`q(y|{\bf x}), q(z|{\bf x}), p({\bf x}|y, z)`)
but differ in their input/output as well as loss functions.
The labelled data has input :math:`({\bf x}, y)` and output :math:`({\bf x'}, y')`.
:math:`y'` corresponds to the predictions from the posterior, while
:math:`{\bf x'}` corresponds to the decoder output.
The loss function is Equation 8 with :math:`\alpha=0.1N` (not the one I derived
in Appendix A).  For the unlabelled case, the input is :math:`{\bf x}` and the output
is the predicted :math:`{\bf x'}`.

For the training, I used the `train_on_batch()` API to train the first network 
on a random batch of labelled data, followed by the second on unlabelled data.
The batches were sized so that the epochs would finish at the same time.
This is not strictly the same as the algorithm from the paper but I'm guessing
it's close enough (also much easier to implement because it's in Keras).
The one cute thing that I did was use vanilla `tqdm` to mimic the `keras_tqdm`
so I could get a nice progress bar.  The latter only works with the regular
`fit` methods so it wasn't very useful.

|h3| Comparison Implementations |h3e|

In the results below I compared a semi-supervised VAE with several other ways
of dealing with semi-supervised learning problems:

* `PCA + SVM`: Here I just ran principal component analysis on the entire image
  set, and then trained a SVM using a PCA-transformed representation on
  only the *labelled* data.
* `CNN`: A vanilla CNN using the Keras `CNN example <https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py>`__
  trained only on *labelled* data.
* `Inception`: Here I used a pre-trained `Inception network <https://keras.io/applications/>`__ available in Keras.
  I pretty much just used the example they had which adds a global average
  pooling layer, a dense layer, followed by a softmax layer.  Trained only on
  the *labelled* data while freezing all the original pre-trained Inception
  layers.  I didn't do any fine-tuning of the Inception layers.

|h2| Semi-supervised Results |h2e|

The datasets I used were MNIST and CIFAR10 with stratified sampling on the
training data to create the semi-supervised dataset.  The test sets are the
ones included with the data.  Here are the results for MNIST:

.. csv-table:: Table 1: MNIST Results
   :header: "Model", "N=100", "N=500", "N=1000", "N=2000", "N=5000"
   :widths: 15, 10, 10, 10, 10, 10

   "PCA + SVM", 0.692, 0.871, 0.891, 0.911, 0.929
   "CNN", 0.262, 0.921, 0.934, 0.955, 0.978
   "M1", 0.628, 0.885, 0.905, 0.921, 0.933
   "M2", "-", "-", 0.975, "-", "-"

The M2 model was only run for :math:`N=1000` (mostly because I didn't really
want to rearrange the code).  From the MNIST results table, we really see the
the M2 model shine where at a comparable sample size, all the other methods
have much lower performance.  You need to get to :math:`N=5000` before the CNN
gets in the same range.  Interestingly at :math:`N=100` the models that make
use of the unlabelled data do better than a CNN which has so little training
data it surely is not learning to generalize.  Next, onto CIFAR 10 results 
shown in Table 2.

.. csv-table:: Table 2: CIFAR10 Results
   :header: "Model", "N=1000", "N=2000", "N=5000", "N=10000", "N=25000"
   :widths: 15, 10, 10, 10, 10, 10

   "CNN", 0.433, 0.4844, 0.610, 0.673, 0.767
   "Inception", 0.661, 0.684, 0.728, 0.751, 0.773
   "PCA + SVM", 0.356, 0.384, 0.420, 0.446, 0.482
   "M1", 0.321, 0.362, 0.375, 0.389, 0.409
   "M2", "0.420", "-", "-", "-", "-"

Again I only train M2 on :math:`N=1000`.  The CIFAR10 results show another
story.  Clearly the pre-trained Inception network is doing the best.  It's
pre-trained on Imagenet which is very similar to CIFAR10.  You have to get to
relatively large sample sizes before even the CNN starts approaching the same
accuracy.  

The M1/M2 results are quite poor, not even beating out PCA in most cases!
My reasoning here is that the CIFAR10 dataset is too complex for the VAE model.
That is, when I look at the images generated from it, it's pretty hard for me
to figure out what the label should be.  Take a look at some of the randomly generated
images from my M2 model:

.. figure:: /images/m2_images.png
  :width: 350px
  :alt: Images generated from M2 VAE model trained on CIFAR data.
  :align: center

  Figure 3: Images generated from M2 VAE model trained on CIFAR data.

Other people have had similar `problems <https://github.com/dojoteef/dvae>`__.
I suspect the :math:`{\bf z}` Gaussian latent variables are not powerful enough
to encode the complexity of the CIFAR10 dataset.  I've read somewhere that the
unimodal nature of the latent variables is thought to be quite limiting, and
here I guess we see that is the case.  I'm pretty sure more recent research has
tried to tackle this problem so I'm excited to explore this phenomenon more
later.

|h2| Conclusion |h2e|

As I've been writing about for the past few posts, I'm a huge fan of scalable
probabilistic models using deep learning.  I think it's both elegant and
intuitive because of the probabilistic formulation.  Unfortunately, VAEs using
Gaussians as the latent variable do have limitations, and obviously they are
not quite the state-of-the-art in generative models (i.e. GANs seem to be the top
dog).  In any case, there is still a lot more recent research in this area that
I'm going to follow up on and hopefully I'll have something to post about soon.
Thanks for reading!


|h2| Further Reading |h2e|

* Previous Posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, `Variational Bayes and The Mean-Field Approximation <link://slug/variational-bayes-and-the-mean-field-approximation>`__, `Maximum Entropy Distributions <link://slug/maximum-entropy-distributions>`__
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
    p_{\theta}({\bf \pi}|y)  &= Dir(\alpha_p + {\bf c}_y) \\
    \tag{A.1}

where :math:`\alpha_p, \alpha_q` are scalar constants, and :math:`{\bf c}_y` is
a vector with 0's and a single 1 representing the categorical observation of :math:`y`.
The latter distribution is just the conjugate prior of a single
observation of a categorical variable :math:`y`, whereas the former
is basically just something we picked out of convenience (remember it's the
posterior approximation that we get to choose).

Let's take a look at the formula for 
`KL divergence between two Dirichlets distributions <http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/>`__
parameterized by vectors :math:`{\bf \alpha}` and :math:`{\bf \beta}`:

.. math::

    KL(p||q) &=  \log \Gamma(\alpha_0) - \sum_{k=1}^{K} \log \Gamma(\alpha_k)
             - \log \Gamma(\beta_0) + \sum_{k=1}^{K} \log \Gamma(\beta_k)
             + \sum_{k=1}^K (\alpha_k - \beta_k)E_{p(x)}[\log x_k] \\
    &=  \log \Gamma(\alpha_0) - \sum_{k=1}^{K} \log \Gamma(\alpha_k)
             - \log \Gamma(\beta_0) + \sum_{k=1}^{K} \log \Gamma(\beta_k)
             + \sum_{k=1}^K (\alpha_k - \beta_k)(\psi(\alpha_k) - \psi(\alpha_0)) \\
    \tag{A.2}

where :math:`\alpha_0=\sum_k \alpha_k` and :math:`\beta_0=\sum_k \beta_k`,
and :math:`\psi` is the `Digamma function <https://en.wikipedia.org/wiki/Digamma_function>`__.

Substituting Equation A.1 into A.2, we have:

.. math::

    KL[q_\phi({\bf \pi}|{\bf x})||p_{\theta}({\bf \pi}|y)] 
            &= \log \Gamma(\alpha_q) 
            - \sum_{k=1}^{K} \log \Gamma(\alpha_q \pi_{\phi,k}({\bf x})) \\
            &\quad - \log \Gamma(K\alpha_p + 1)
                + \sum_{k=1}^{K} \log \Gamma(\alpha_p + c_{y,k}) \\
            &\quad + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - \alpha_p - {\bf c}_{y,k})
               (\psi(\alpha_{q,k}\pi_{\phi,k}) - \psi(\alpha_q)) \\
    &= K_2 
        - \sum_{k=1}^{K} \log \Gamma(\alpha_q \pi_{\phi,k}({\bf x}))
        + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - \alpha_p - {\bf c}_{y,k})
               (\psi(\alpha_q\pi_{\phi,k}) - \psi(\alpha_q)) \\
    \tag{A.3}

Here, most of the Gamma functions are just constants so we can absorb them into a constant.
Okay, here's where it gets a bit hand wavy (it's the only way I could figure out how to simplify
the equation to what it had in the paper).
We're going to pick a big :math:`\alpha_q` and a small :math:`\alpha_p`.  Both
are hyper parameters so we can freely do as we wish.  With this assumption, we're going to
progressively simplify and approximate Equation A.3:

.. math::

    &KL[q_\phi({\bf \pi}|{\bf x})||p_{\theta}({\bf \pi}|y)] \\
    &= K_2 
        - \sum_{k=1}^{K} \log \Gamma(\alpha_q \pi_{\phi,k}({\bf x}))
        + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - \alpha_p - {\bf c}_{y,k})
               (\psi(\alpha_q\pi_{\phi,k}) - \psi(\alpha_q)) \\
    &\leq K_3 
        + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - \alpha_p - {\bf c}_{y,k})
               (\psi(\alpha_q\pi_{\phi,k}({\bf x})) - \psi(\alpha_q)) \\
    &\approx K_3 
        + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - {\bf c}_{y,k})
               (\psi(\alpha_q\pi_{\phi,k}({\bf x})) - \psi(\alpha_q)) \\
    &= K_4 
        + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - {\bf c}_{y,k})
               \psi(\alpha_q\pi_{\phi,k}({\bf x})) \\
    &\approx K_4 
        + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - {\bf c}_{y,k})
               \log(\alpha_q\pi_{\phi,k}({\bf x})) \\
    &\leq K_5 
        + \sum_{k=1}^K (\alpha_q\pi_{\phi,k}({\bf x}) - {\bf c}_{y,k})
               \log(\pi_{\phi,k}({\bf x})) \\
    &\leq K_5 
        + \alpha_q \sum_{k=1}^K \pi_{\phi,k}({\bf x})\log(\pi_{\phi,k}({\bf x}))
        - \sum_{k=1}^K {\bf c}_{y,k} \log(\pi_{\phi,k}({\bf x})) \\
    &= K_5 - \alpha_q H(q)
        - \sum_{k=1}^K  \log(\pi_{\phi,k}({\bf x})^{{\bf c}_{y,k}}) \\
    &= K_5 
        - \alpha_q H(q) - \log(q(y|{\bf x})) \\
    &\leq K_5 - \log(q(y|{\bf x})) \\
    \tag{A.4}

This is quite a mouthful to explain since I'm just basically waving my hand
to get to the final expression.  First, we drop the Gamma function
in the second term and upper bound it by a new constant :math:`K_3` because our
:math:`\alpha_q` is large, its the gamma function is always positive.
Next, we drop :math:`\alpha_p` since it's small (let's just make it arbitrarily
small).  We then drop :math:`\psi(\alpha_q)`, a constant, because when we
expand it out we get a constant (recall :math:`\sum_{k=1}^K \pi_{\phi, k}({\bf x}) = 1`).

Now we're getting somewhere!  Since :math:`\alpha_q` is again large the
`Digamma function <https://en.wikipedia.org/wiki/Digamma_function>`__ 
is upper bounded by :math:`\log(x)` when :math:`x>0.5`, so we'll just make
this substitution.  Finally, we get something that looks about right.
We just rearrange a bit and two non-constant terms involving entropy of
:math:`q` and the probability of a categorical variable with parameter
:math:`\pi({\bf x})`.  We just upper bound the expression by dropping
the :math:`-H(q)` term since entropy is always positive to get us to
our final term :math:`-\log(q(y|{\bf x}))` that Kingma put in his paper.
Although, one thing I couldn't quite get to is the additional constant :math:`\alpha`
that is in front of :math:`\log(q(y|{\bf x}))`.

Admittedly, it's not quite precise, but it's the only way I figured out how to
derive his expression without just arbitrarily adding an extra term to the
loss function (why work out any math when you're going to arbitrarily add
things to the loss function?).  Please let me know if you have a better way of
deriving this equation.



|br|
