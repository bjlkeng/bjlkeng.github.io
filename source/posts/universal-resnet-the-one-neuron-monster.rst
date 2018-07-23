.. title: Universal Resnet: The One-Neuron Monster
.. slug: universal-resnet-the-one-neuron-monster
.. date: 2018-07-23 08:03:28 UTC-04:00
.. tags: ResNet, residual networks, hidden layers, neural networks, universal approximator, mathjax
.. category: 
.. link: 
.. description: Some fun playing around with neural network universal approximators.
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

.. |hr| raw:: html

   <hr>


This post is going to talk about a paper I read recently called *ResNet with
one-neuron hidden layers is a Universal Approximator*.  It talks about a simplified
Residual Network as a universal approximator, giving some theoretical backing to 
the wildly successful ResNet architecture.  Of course, I'm not going to go through
the any of the theoretical stuff, instead I'm going to play around to see if we
can get close to these theoretical limits.

(You might also want to checkout my previous post where I played around with
ResNets: `Residual Networks <link://slug/residual-networks>`__)


|H2| Universal Approximation Theorem |H2e|

The `Universal Approximation Theorem
<https://en.wikipedia.org/wiki/Universal_approximation_theorem>`__ is one of
the most well known (and often incorrectly used) theorems when it comes to
neural networks.  It's probably because neural networks don't have as much
theoretical depth as other techniques.  I'll paraphrase the theorem here
(if you're really interested in the math click on the Wikipedia link):

    The **universal approximation theorem** states that a feed-forward
    networks with a single hidden layer containing a finite number of neurons
    can approximate functions on compact sets on :math:`\mathbb{R}^n`, under
    mild assumptions of the activation function (nonconstant,
    monotonically-increasing, continuous).

The implications is *theoretically*, we can fit any function just by increasing
the width of a single hidden layer arbitrarily large.  Of course, we know *"In theory,
theory and practice are the same; in practice, they are not."*

For me, this theorem doesn't strike me as all that insightful (or useful).
First, in practice we never arbitrarily make the width of a neural network
really wide, never mind having a single hidden layer.  Second, it seems
intuitive that this might be the case doesn't it?  If we think about approximating
a function with an arbitrary number of `piecewise linear functions <https://en.wikipedia.org/wiki/Piecewise_linear_function>`__, then we should be able to approximate
*most* well-behaved functions (that's basically how derivation works).
Similarly, if we have a neuron for each "piece" of the function, we should
intuitively be able to approximate it to any degree (although I'm sure the
actual proof is more complicated and elegant).

.. figure:: /images/piecewise_linear_approximation.png
  :height: 300px
  :alt: Piecewise Linear Function
  :align: center

  Figure 1: If we can approximate functions with an arbitrary number of
  piecewise lienar functions, why can't we do it with an arbitrary number of
  hidden units? (source: Wikipedia)

So practically, there isn't much that's interesting about this (basic) version
of the theorem.  And definitely, people should stop quoting it as if it somehow
"proved" something.

|H2| Universal Approximation Theorem for Width-Bounded ReLU Networks |H2e|

A *much* more interesting result for neural networks is a universal approximation
theorem for width-bounded neural networks published recently (2017) from [2]:

    Let :math:`n` denotes the input dimension, we show that width-:math:`(n + 4)` 
    ReLU networks can approximate any Lebesgue integrable function on
    n-dimensional space with respect to L1 distance.

    Additionally, except for a negligible set, most functions cannot
    be approximated by a ReLU network of width at most :math:`n`.

So basically if you have just 4 extra neurons in each layer and arbitrary
number of hidden layers, you can approximate any Lebesque integrable function
(which are most functions).  This result is very interesting because this 
is starting to look like a feed-forward network that we might build.  Just take
the number of inputs, add 4, and then make a bunch of hidden layers of that
size.  Of course, training this is another story, which we'll see below.
The negative result is also very interesting.  We need some extra width or else
we'll never be able to approximate anything.

There are also a whole bunch of other results.  See the references from [1],
[2] if you're interested.

|H2| Universal Approximation Theorem for The One-Neuron Resnet |H2e|




|h2| Further Reading |h2e|

* Previous posts: `Residual Networks <link://slug/residual-networks>`__
* Wikipedia: `Universal Approximation Theorem <https://en.wikipedia.org/wiki/Universal_approximation_theorem>`__
* [1] `ResNet with one-neuron hidden layers is a Universal Approximator <https://arxiv.org/abs/1806.10909>`__, Hongzhou Lin, Stefanie Jegelka
* [2] `The Expressive Power of Neural Networks: A View from the Width <https://arxiv.org/abs/1709.02540>`__ Zhou Lu, Hongming Pu, Feicheng Wang, Zhiqiang Hu, Liwei Wang
