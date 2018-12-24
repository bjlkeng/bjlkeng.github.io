.. title: Importance Sampling and Estimating Marginal Likelihood in Variational Autoencoders
.. slug: importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders
.. date: 2018-09-25 08:20:11 UTC-04:00
.. tags: variational calculus, autoencoders, importance sampling, generative models, MNIST, autoregressive, CIFAR10, Monte Carlo, mathjax
.. category: 
.. link: 
.. description: A short post describing how to use importance sampling to estimate marginal likelihood in variational autoencoers.
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

It took a while but I'm back!  This post is kind of a digression (which seems
to happen a lot) along my journey of learning more about probabilistic
generative models.  There's so much in ML that you can't help learning a lot
of random things along the way.  That's why it's interesting, right?

Today's topic is *importance sampling*.  It's a really old idea that you may
have learned in a statistics class (I didn't) but somehow in deep learning,
what's old is new right?  How this is relevant to the discussion is that when
we have a model without an explicit likelihood function (e.g. a variational
autoencoder), we still want to be able to estimate the marginal likelihood
given the data.  It's kind of a throwaway line in the experiments of some VAE
papers when comparing different models.  I was curious how it was computed and
it took me down this rabbit hole.  Turns out it's actually pretty interesting!
As usual, I'll have a mix of background material, examples, math and code 
to build some good intuition around this topic.  Enjoy!

.. TEASER_END

|h2| A Brief Review of Monte Carlo Simulation |h2e|

`Monto Carlo simulation <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__
methods are a broad class of algorithms that use repeated sampling 
(hence Monte Carlo like the casino in Monaco) to obtain a numerical result.
These techniques are useful when we cannot explicitly compute the end result.
The simplest example is computing an expectation when the closed form result is
unavailable but we can sample from the underlying distribution (there are many
others examples, see the Wiki page).
In this case, we can take our usual equation for expectation and
approximate it by a summation.  Given random variable :math:`X` with density
function :math:`p(x)`, distributed according to :math:`Q`, we have:

.. math::

    E(X) &= \int x \cdot p(x) dx \\
         &\approx \frac{1}{n} \sum_{i=1}^n X_i && \text{where } X_i \sim Q \\
    \tag{1}

This is a simple restatement of the 
`law of large numbers <https://en.wikipedia.org/wiki/Law_of_large_numbers>`__.
To make this a bit more useful, we don't just want the expectation of a single
random variable, instead we usually have some (deterministic) function of a
vector random variables.  Using the same idea as Equation 1, we have:

.. math::

    E(f({\bf X})) &= \int f({\bf x}) p({\bf x}) d{\bf x} \\
         &\approx \frac{1}{n} \sum_{i=1}^n f({\bf X_i}) && \text{where } {\bf X_i} \sim {\bf Q} \\
    \tag{2}

where all of the quantities are now vectors and :math:`f` is a deterministic
function.  
For more well behaved smaller problems, we can get a reasonably good
estimate of this expectation with :math:`\frac{1}{n}` convergence 
(by the `central limit theorem <https://en.wikipedia.org/wiki/Central_limit_theorem>`__).
That is, quadrupling the number of points halves the error.  Let's take a look
at an example.


.. admonition:: Example 1: Computing the expected number of times to miss a
    project deadline (source [1])

    .. figure:: /images/task_dag.png
      :height: 300px
      :alt: DAG of Tasks
      :align: center
    
      Figure 1: The graph of task dependencies (source [1]).

    Imagine, we're running a project with 10 distinct steps.  The project
    has dependencies shown in Figure 1.  Further, the mean time to 
    complete each task is listed in Table 1.

    .. csv-table:: Table 1: Mean Task Times
       :header: "Task j", "Predecessors", "Duration (days)"
       :widths: 15, 10, 10
       :align: center
    
       "1", None, 4
       "2", 1, 4
       "3", 1, 2
       "4", 2, 5
       "5", 2, 2
       "6", 3, 3
       "7", 3, 2
       "8", 3, 3
       "9", "5,6,7", 2
       "10", "4, 8, 9", 2
    
    If we add up the critical path in the graph we get a completion time of 15
    days.  But estimating each task completion time as a point estimate is not
    very useful when we want to understand if the project is at risk of delays.
    So let's model each task as an independent random `exponential distribution
    <https://en.wikipedia.org/wiki/Exponential_distribution>`__ with mean
    according to the duration of the task.  When we simulate this our 
    mean time to completion is around :math:`18.2`.  (It's not exactly the 15 we
    might expect by adding up the critical path because the 
    `sum of two exponentials <https://math.stackexchange.com/questions/474775/sum-of-two-independent-exponential-distributions>`__
    is not a simple exponential distribution.)


    This example along with the one below is shown in the notebook
    **TODO{FIX ME}**.


    

|h2| Importance Sampling |h2e|




|h2| Estimating Marginal Likelihood in Variational Autoencoders |h2e|



|h2| Implementaiton Details |h2e|



* Getting the loss function right was hard
* Constrain the sigmas for Z and for s
* batch norm is kind of important for training deep networks...
* RELU + batch_norm (no weight norm) for some reason totally bombed, switching to ELU changed things...
* Each iteration of the code I seem to clean it up a bit more... it's just some hacking, no need to be clean


|h2| Conclusion |h2e|

|h2| Further Reading |h2e|

* [1] "Importance Sampling", Art Owen, `<https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf>`__
* [2] "Variational Inference with Normalizing Flows", Danilo Jimenez Rezende, Shakir Mohamed, `ICML 2015 <https://arxiv.org/abs/1505.05770>`__
* [3] "Pixel Recurrent Neural Networks", Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu, `<https://arxiv.org/pdf/1601.06759.pdf>`__

* Wikipedia: `Importance Sampling <https://en.wikipedia.org/wiki/Importance_sampling>`__, `Monto Carlo methods <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__

* Previous posts: `Variational Autoencoders <link://slug/variational-autoencoders>`__, `A Variational Autoencoder on the SVHN dataset <link://slug/a-variational-autoencoder-on-the-svnh-dataset>`__, `Semi-supervised Learning with Variational Autoencoders <link://slug/semi-supervised-learning-with-variational-autoencoders>`__, `Autoregressive Autoencoders <link://slug/autoregressive-autoencoders>`__, `Variational Autoencoders with Inverse Autoregressive Flows <link://slug/variational-autoencoders-with-inverse-autoregressive-flows>`__


