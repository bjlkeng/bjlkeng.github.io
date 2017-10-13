.. title: Autoregressive Autoencoders
.. slug: autoregressive-autoencoders
.. date: 2017-10-05 08:14:15 UTC-04:00
.. tags: autoencoders, autoregressive, generative models, MADE, MNIST, mathjax
.. category: 
.. link: 
.. description: A write up on Masked Autoencoder for Distribution Estimation (MADE).
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


You might think that I'd be bored with autoencoders by now but I still
find them extremely interesting!  In this post, I'm going to be explaining
a cute little idea that I came across in the paper `MADE: Masked Autoencoder
for Distribution Estimation <https://arxiv.org/pdf/1502.03509.pdf>`_.
Traditional autoencoders are great because they can perform unsupervised
learning by mapping an input to a latent representation.  However, one
drawback is that they don't have a solid probabilistic basis
(of course there are other variants of autoencoders that do, see previous posts
`here <link://slug/variational-autoencoders>`__, 
`here <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, and
`here <link://slug/semi-supervised-learning-with-variational-autoencoders>`__). 
By using what the authors define as the *autoregressive property*, we can
transform the traditional autoencoder approach into a fully probabilistic model
with very little modification! As usual, I'll provide some intuition, math and
an implementation.

.. TEASER_END

|h2| Vanilla Autoencoders |h2e|

The basic `autoencoder <https://en.wikipedia.org/wiki/Autoencoder>`_
is a pretty simple idea.  Our primary goal is take an input sample
:math:`x` and transform it to some latent dimension :math:`z` (*encoder*),
which hopefully is a good representation of the original data.  As
usual, we need to ask ourselves: what makes a good representation?  A vanilla
autoencoder's answer: "*A good representation is one where you can reconstruct
the original input!*".  The process of reconstruction from the latent
dimension :math:`z` back to our reconstruction of the input sample
:math:`\hat{x}` is called the *decoder*.  Figure 1 shows a picture of what this
looks like.

.. figure:: /images/autoencoder_structure.png
  :width: 400px
  :alt: Vanilla Autoencoder
  :align: center

  Figure 1: Vanilla Autoencoder (source: `Wikipedia <https://en.wikipedia.org/wiki/Autoencoder>`_)

From Figure 1, we typically will use a neural network as the encoder and
a different (usually similar) neural network as the decoder.  Additionally,
we'll typically put a sensible loss function on the output to ensure :math:`x`
and :math:`\hat{x}` are as close as possible:

.. math::

    \mathcal{L_{\text{binary}}}({\bf x}) &= \sum_{i=1}^D -x_i\log \hat{x}_i - (1-x_i)\log(1-\hat{x_i}) \tag{1} \\
    \mathcal{L_{\text{real}}}({\bf x}) &= \sum_{i=1}^D  (x_i - \hat{x}_i)^2 \tag{2}

Here we assume that our data point :math:`{\bf x}` has :math:`D` dimensions.
The loss function we use will depend on the form of the data.  For binary data,
we'll use the cross entropy and for real-valued data we'll use the mean squared
error.  These correspond to modelling :math:`x` as a Bernoulli and Gaussian
respectively (see the box).

.. admonition:: Negative Log-Likelihoods (NLL) and Loss Functions

    The loss functions we typically use in training machine learning models are
    usually derived by an assumption on the probability distribution of each
    data point (typically assuming identically, independently distributed (IID)
    data).  It just doesn't look that way because we typically use the negative
    log-likelihood as the loss function.  We can do this because we're usually
    just looking for a point estimate (i.e. optimizing) so we don't need to
    worry about the entire distribution, just a single point that gives us the
    highest probability.

    For example, if our data is binary, then we can model it as a 
    `Bernoulli <https://en.wikipedia.org/wiki/Bernoulli_distribution>`__ 
    with parameter :math:`p` on the interval :math:`(0,1)`.  The probability
    of seeing a given 0/1 :math:`x` value is then:

    .. math::

        P(x) = p^x(1-p)^{(1-x)}  \tag{3}
    
    If we take the logarithm and negate it, we get the binary cross entropy
    loss function:

    .. math::

        \mathcal{L_{\text{binary}}}(x) = -x\log p - (1-x)\log(1-p) \tag{4}

    This is precisely the expression from Equation 1, except we replace
    :math:`x=x_i` and :math:`p=\hat{x_i}`, where the former is the observed
    data and latter is the estimate of the parameters that our model gives. 

    Similarly, we can do the same trick with a 
    `normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`__.
    Given an observed real-valued data point :math:`x`, the probability density
    for parameters :math:`\mu, \sigma^2` is given by:

    .. math::
        p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \tag{5}

    Taking the negative logarithm of this function, we get:

    .. math::

        -\log p(x) = 
        \frac{1}{2}\log(2\pi \sigma^2) + \frac{1}{2\sigma^2} (x-\mu)^2 \tag{6}

    Now if we assume that the variance is the same fixed value for all our data
    points, then the only parameter we're optimizing for is :math:`\mu`. So 
    adding and multiplying by a bunch of constants to our main expression
    doesn't change the optimal (highest probability) point so we can just
    simplify it (when optimizing) and still get the same point solution:

    .. math::

        \underset{\mu}{\operatorname{argmax}} -\log p(x) = 
        \underset{\mu}{\operatorname{argmax}} \mathcal{L_{\text{real}}}(x) =
        \underset{\mu}{\operatorname{argmax}} (x-\mu)^2
        \\ \tag{7}

    Here our observation is :math:`x` and our model would produce an
    estimate of the parameter :math:`\mu` i.e. :math:`\hat{x}` in this case.  I
    have some more details on this in one of my previous posts on
    `regularization <link://slug/probabilistic-interpretation-of-regularization>`__.
   
|h3| Losing Your Identity |h3e|

Now this is all well and good but an astute observer will notice that unless we
put some additional constraints, our autoencoder can just set :math:`z=x` (i.e.
the identity function) and generate a perfect reconstruction.  What better
representation for a reconstruction than *exactly* the original data?  This is
generally solved by making it difficult to learn just the identity function.

The easiest method is to just make the dimensions of :math:`z` smaller than
:math:`x`.  For example, if your image has 900 pixels (30 x 30) then make the
dimensions of :math:`z`, say 100.  In this way, you're "forcing" the
autoencoder to learn a more compact representation.

Another method used in *denoising autoencoders* is to artificially introduce
noise on the input :math:`x' = \text{noise}(x)` (e.g. Gaussian noise) but still
compare the output of the decoder with the clean value of :math:`x`.  The
intuition here is that a good representation is robust to any noise that you
might give it.  Again, this prevents the autoencoder from just learning the
identify mapping (because your input is not the same as your output anymore).

In both cases, you will eventually end up with a pretty good latent representation
of :math:`x` that can be used in all sorts of applications such as 
`semi-supervised learning <link://slug/semi-supervised-learning-with-variational-autoencoders>`__.

|h3| Proper Probability Distributions |h3e|

Although vanilla autoencoders do pretty well in learning a latent
representation of data in an unsupervised manner, they don't have a proper
probabilistic interpretation.  We put a loss function on the outputs of the
autoencoder in Equation 1 and 2 but that doesn't automatically mean our
autoencoder will generate a proper distribution of the data!  Let me explain.

Ideally, we would like the unsupervised autoencoder to learn the distribution
of the data.  That is, for each one of our :math:`\bf x` values, we would like
to be able to evaluate the probability :math:`P({\bf x})` to see how often we
would expect to see this data point.  Implicitly this means that if we sum over
all *possible* :math:`x` values, we should get :math:`1`, 
i.e. :math:`\sum_x P(x) = 1`.  For autoencoders, we can show that this property
is not guaranteed.  

Consider two samples :math:`x_1`, and :math:`x_2`.  Let's say (regardless of
what type of autoencoder we use), our neural network "memorizes" these two
samples and is able to reconstruct them perfectly.  That is, 
pass :math:`x_1` into the autoencoder and get *exactly* :math:`x_1`; 
pass :math:`x_2` into the autoencoder and get *exactly* :math:`x_2`.  
If this happened, it would be a good thing (as long as we had a bottleneck or a
denoising autoencoder) because we have a learned a really powerful latent
representation that can reconstruct the data perfectly!
However, his implies the loss from Equation 1 (or 2 in the continuous case) is
:math:`0`.  If we negative take the exponential to translate it to a
probability this means both :math:`P(x_1)=1` and :math:`P(x_0)=1`, which of
course is not a valid probability distribution.

In contrast, if our model does model the data distribution properly, then we
end up with a fully 
`generative model <https://en.wikipedia.org/wiki/Generative_model>`__,
where you can usually do nice things like sample from it (e.g. generate new
images).

For vanilla autoencoders, we started with some neural network and then tried to
apply some sort of probabilistic interpretation that doesn't quite work out.  I
like it the other way around: start with a probabilistic model and then figure
out how to use neural networks to help you add more capacity and scale it.


|h2| Autoregressive Autoencoders |h2e|

So vanilla autoencoders don't quite get us to a proper probability distribution
of the data, but is there a way to modify them to get us there?  Let's review
the `product rule <https://en.wikipedia.org/wiki/Chain_rule_(probability)>`__:

.. math::

    p({\bf x}) = \prod_{i=1}^{D} p(x_i | {\bf x}_{<D})  \tag{8}

where :math:`{\bf x}_{<D} = [x_1, \ldots, x_{i-1}]`.  Basically, component
:math:`i` of :math:`{\bf x}` only depends on the dimensions of :math:`j < i`.

So how does this help us? In vanilla autoencoders, each component :math:`x_i`
could depend on any of its components :math:`x_1,\ldots,x_n`, this resulted an
improper probability distribution.  If we start with the product rule, which
guarantees a proper distribution, we can work backwards to map the autoencoder
to this model.

For example, let's consider binary data (say a binarized image).  :math:`x_1`
does not depend on any other components of :math:`{\bf x}`, therefore our
implementation should just need to estimate a single parameter :math:`p_1` for
this pixel.  
How about :math:`x_2` though?  Now we let :math:`x_2` depend *only* on
:math:`x_1` since we have :math:`p(x_2|x_1)`.  This dependency can be modelled
using a non-linear function say maybe a neural network? So we'll have
some neural net that maps :math:`x_1` to the :math:`x_2` output.  Now consider
the general case of :math:`x_j`,  we can have a neural net that maps 
:math:`\bf x_{<j}` to the :math:`x_j` output.  Lastly, there's no reason that
each step needs to be a separate neural network, we can just put it all
together in a single shared neural network so long as we follow a couple of rules:

1. Each output of the network :math:`\hat{x}_i` represents the probability
   distribution :math:`p(x_i|{\bf x_{<i}})`.
2. Each output can only have connections (recursively) to smaller indexed
   inputs :math:`\bf x_{<i}` and not any of the other ones.

Said another way, our neural net first learns :math:`p(x_1)` (just a single
parameter value in the case of binarized data), then iteratively learns the
function mapping from :math:`{\bf x_{<j}}` to :math:`x_j`.  In this view of the
autoencoder, we are sequentially predicting (i.e. regressing) each dimension of
the data using its previous values, hence this is called the *autoregressive*
property of autoencoders.

Now that we have a fully probabilistic model that uses autoencoders, let's
figure out how to implement it!

|h3| Masks and the Autoregressive Network Structure |h3e|

The autoregressive autoencoder is referred to as a "Masked Autoencoder for
Distribution Estimation", or MADE.  "Masked" as we shall see below and
"Distribution Estimation" because we now have a fully probabilistic model.
("Autoencoder" now is a bit looser because we don't really have a concept of
encoder and decoder anymore, only the fact that the data is put on the
input/output.)

From the autoregressive property, all we want to do is ensure that we only have
connections (recursively) from input :math:`i` to output :math:`j` where
:math:`i < j`.  One way to accomplish this is to not make these connections in
the first place, but that's a bit annoying because we can't easily use our
existing infrastructure for neural networks.  

The main observation here is that a connection with weight :math:`0` is the
same as no connection at all.  So all we have to do is zero-out the connections
we don't want.  We can do that easily with a "mask" for each weight matrix
which says which connections we want and which we don't.

This is a simple modification to our standard neural networks.  Consider a one
hidden layer autoencoder with input :math:`x`:

.. math::

    {\bf h}({\bf x}) &= {\bf g}({\bf b} + {\bf (W \odot M^W)x}) \\
    {\hat{\bf x}} &= \text{sigm}({\bf c} + {\bf (V \odot M^V)h(x)})  \tag{9}

where:

* :math:`\odot` is an element wise product
* :math:`\bf x, \hat{x}` is our vectors of input/output respectively
* :math:`\bf h(x)` is the hidden layer
* :math:`\bf g(\cdot)` is the activation function of the hidden layer
* :math:`\text{sigm}(\cdot)` is the sigmoid activation function of the output layer
* :math:`\bf b, c` are the constant biases for the hidden/output layer respectively
* :math:`\bf W, V` are the weight matrices for the hidden/output layer respectively
* :math:`\bf M^W, M^V` are the weight mask matrices for the hidden/output layer respectively


So long as our masks are set such that the autoregressive property is
satisfied, the network can produce a proper probability distribution.
One subtlety here is that for each hidden unit, we need to define an index that
says which inputs it can be connected to (which also determines which
index/output in the next layer it can be connected to).  We'll use the notation
in the paper of :math:`m^l(k)` to denote the index assigned to hidden node 
:math:`k` in layer :math:`l`.  Our general rule for our masks is then:

.. math::

    M^{W^l}_{k', k} = \left\{
                \begin{array}{ll}
                  1 \text{ if } m^l(k') \geq m^{l-1}(k)  \\
                  0 \text{ otherwise}
                \end{array}
              \right. \\ \tag{10}

Basically, for a given node, only connect it to nodes that have an index less
than or equal to it's index.  This will guarantee that a given index will
recursively obey our auto-regressive property.

The output mask has a slightly different rule:

.. math::

    M^{V}_{d, k} = \left\{
                \begin{array}{ll}
                  1 \text{ if } d > m^{L}(k)  \\
                  0 \text{ otherwise}
                \end{array}
              \right.  \\ \tag{11}

which replaces the less than equal with just an equal.  This is important
because the first node should not depend on any other ones so it should not
have any connections (will only have the bias connection), and the last node
can have connections (recursively) to every other node except its respective
input.

Finally, one last topic to discuss is how to assign :math:`m^l(k)`.  It doesn't
really matter too much as long as you have enough connections for each index.
The paper did a natural thing and just sampled from a uniform distribution
with range :math:`[1, D-1]`.  Why only up to :math:`D-1`?  Recall, we should
never assign index :math:`D` because it will never be used so there's no use
in connecting anything to :math:`D` (nothing can ever depend on the
:math:`D^{\text{th}}` input).  Figure 2 (from the original paper) shows this
whole process pictorially.

.. figure:: /images/made_mask.png
  :width: 450px
  :alt: MADE Masks
  :align: center

  Figure 2: MADE Masks (Source: `[1] <https://arxiv.org/pdf/1502.03509.pdf>`__)

A few things to notice:

* Output 1 is not connected to anything.  It will just be estimated with a
  single constant parameter derived from the bias node.
* Input 3 is not connected to anything because no node should depend on it
  (autoregressive property). 
* :math:`m^l(k)` are more or less assigned randomly.
* If you trace back from output to input, you will see that the autoregressive
  property is maintained.

So then implementing MADE is as simple as providing a weight mask
and doing an extra element-wise product. Pretty simple, right?

|h3| Ordering Inputs, Masks, and Direct Connections |h3e|

A few other minor topics that can improve the performance of the MADE.  The
first is the ordering of the inputs.  We've been taking about "Input 1, 2, 3,
..." but usually there is no natural ordering of the inputs.  We can
arbitrarily pick any ordering that we want just by shuffling :math:`{\bf m^0}`,
the selection layer for the input.  This can even be performed at each
mini-batch to get an "average" over many different models.

The next idea is also very similar, instead of just resampling the input
selection, resample all :math:`{\bf m^L}` selections.  In the paper, they
mention the best results are having a fixed number of configurations for these
selections (and their corresponding masks) and rotating through them in the
mini-batch training.

The last idea is just to add a direct connection path from input to output
like so:

.. math::

    {\hat{\bf x}} = \text{sigm}\big({\bf c} + {\bf (V \odot M^V)h(x)}\big)
                    + \big({\bf A} \odot {\bf M^A}\big){\bf x}  \tag{12}

where :math:`{\bf A}` is the weight matrix that directly connects inputs to outputs,
and :math:`{\bf M^A}` is the corresponding mask matrix that follows the
autoregressive property.

|h3| Generating New Samples |h3e|

One final idea that isn't explicitly mentioned in the paper is how to generate
new samples.  Remember, we now have a fully generative probabilistic model for
our autoencoder.  It turns out it's quite easy but a bit slow.  The main idea
(for binary data):

1. Randomly generate vector :math:`{\bf x}`, set :math:`i=1`.
2. Feed :math:`{\bf x}` into autoencoder and generate outputs 
   :math:`\hat{\bf x}` for the network, set :math:`p=x_i`.
3. Sample from a Bernoulli distribution with parameter :math:`p`, set
   :math:`x_{i}=\text{Bernoulli}(p)`.
4. Increment :math:`i` and repeat steps 2-4 until `i > D`. 

Basically, we're iteratively calculating :math:`p(x_i|{\bf x_{1,\ldots,i-1}})`
by doing a forward pass on the autoencoder each time.  Along the way, we sample
from the Bernoulli distribution and feed the sampled value back into the
autoencoder to compute the next parameter for the next bit.
It's a bit inefficient but MADE is also a relatively small modification to the
vanilla autoencoder so you can't ask for too much.

|h2| MADE Implementation |h2e|

I implemented a MADE layer and built a network using a binarized MNIST dataset
similar to what they used in the original paper
(`notebook <https://github.com/bjlkeng/sandbox/blob/master/notebooks/masked_autoencoders/made-mnist.ipynb>`__).

My implementation is a lot simpler than the one used in the paper.  I used
Keras and created a custom "MADE" layer that took as input the number of layers,
number of hidden units per layer, whether or not to randomize the input
selection, as well as standard stuff like dropout and activation function.
I didn't implement any of the randomized masks for minibatchs because it was
a bit of a pain.  I did implement the direct connection though.

*(As an aside: I'm really a big fan of higher-level frameworks like Keras, 
it's quite wonderful.  The main reason is that for most things I have the nice
Keras frontend, and then occasionally I can dip down into the underlying
primitives when needed via the Keras "backend".  I suspect when I eventually
get around to playing with RNNs it's not going to be as wonderful but for now
I quite like it.)*

I was able to generate some new digits that are not very pretty, shown 
in Figure 3.

.. figure:: /images/mnist-made.png
  :width: 400px
  :alt: Generated MNIST images using Autoregressive Autoencoder
  :align: center

  Figure 3: Generated MNIST images using Autoregressive Autoencoder

It's a bit hard to make out any numbers here.  If you squint hard enough, you
can make out some "4"s,  "3"s, "6"s, maybe some "9"s?  The ones in the paper look
a lot better (although still not perfect, there were definitely some that were
hard to make out).  

The other thing is that I didn't use their exact version of binarized MNIST,
I just took the one from Keras and did a `round()` on each pixel.  This might
also explain why I was unable to get as good of a negative log-likelihood as
them.  In the paper they report values :math:`< 90` (even with a single mask)
but the lowest I was able to get on my test set was around :math:`99`, and that
was after a bunch of tries tweaking the batch and learning rate (more typical
was around :math:`120`).  It could be that their test set was easier, or the
fact that they did some hyper-parameter tuning for each experiment, whereas I
just did some trial and error tuning.

|h3| Implementation Notes |h3e|

Here are some random notes that I came across when building this MADE:

* Adding a direct (auto-regressive) connection between inputs and outputs
  seemed to make a huge difference (150 vs. < 100 loss).  For me, this
  basically was the make-or-break piece for implementing a MADE.  It's funny that
  it's just a throw-away paragraph in the actual paper.  Probably because the
  idea was from an earlier paper in 2000 and not the main contribution of the
  paper.  For some things, you really have to implement it to understand the
  important parts, papers don't tell the whole story!
* I had to be quite careful when coding up layers since getting the indexes for
  selection exactly right is important.  I had a few false starts because I
  mixed up the indexes.  When using the high-level Keras API, there's not much
  of this detailed work, but when implementing your own layers it's important!
* I tried a random ordering (just a single one for the entire training, not 
  one per batch) and it didn't really seem to do much.
* In their actual implementation, they also add dropout to all their layers.  I
  added it too but didn't play around with it much except to try to tune it to
  get a lower NLL.  One curious thing I found out was about using the
  `set_learning_phase() <https://keras.io/backend/>`__ API.  When implementing
  dropout, I basically just took the code from the dropout layer and inserted
  into my custom layer.  However, I kept getting an error, it turns out that
  I had to use `set_learning_phase(1)` during training, and
  `set_learning_phase(0)` during prediction because the Keras dropout
  implementation uses `in_train_phase(<train_input>, <test_input>)`, which
  switches between two behaviours for training/testing based on the status of
  this bit.  For some reason when using the regular dropout layer you don't
  have to do this but when doing it in a custom layer you do?  I suspect I
  missed something in my custom layer that happens in the dropout layer.

|h2| Conclusion |h2e|

So yet *another* post on autoencoders, I can't seem to get enough of them!
Actually I still find them quite fascinating, which is why I'm following this
line of research about fully probabilistic generative models.  There's still at
least one or two more papers in this area that I'm really excited to dig into
(at which point I'll have approached the latest published work), so expect more
to come!


|h2| Further Reading |h2e|

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, `Semi-supervised Learning with Variational Autoencoders <link://slug/semi-supervised-learning-with-variational-autoencoders>`__
* My implementation on Github: `notebook <https://github.com/bjlkeng/sandbox/blob/master/notebooks/masked_autoencoders/made-mnist.ipynb>`__
* [1] "MADE: Masked Autoencoder for Distribution Estimation", Germain, Gregor, Murray, Larochelle, `ICML 2015 <https://arxiv.org/pdf/1502.03509.pdf>`_
* Wikipedia: `Autoencoder <https://en.wikipedia.org/wiki/Autoencoder>`_
* Github code for "MADE: Masked Autoencoder for Distribution Estimation", https://github.com/mgermain/MADE

