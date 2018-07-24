.. title: Universal ResNet: The One-Neuron Monster
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


|H2| Universal Approximation Theorems |H2e|

|H3| The OG Universal Approximation Theorem |H3e|

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

|H3| Universal Approximation Theorem for Width-Bounded ReLU Networks |H3e|

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
There are also a whole bunch of other results on neural networks that have been
proven.  See the references from [1] and [2] if you're interested.

|H3| Universal Approximation Theorem for The One-Neuron ResNet |H3e|

ResNet is a neural network architecture that contains a "residual connection".
Basically, it provides a shortcut from layer :math:`i` merging it with addition
to layer :math:`i+k` for some constant :math:`k`.  In between, you can do all
kinds of interesting things such as have multiple layers but the important
thing is that the input width of layer :math:`i` is the same as the output
width of layer :math:`i+k`.  Figure 2 shows a simplified ResNet architecture
where the "in between" transformation is a single neuron.

.. figure:: /images/basic_resnet.png
  :height: 200px
  :alt: Basic ResNet Block
  :align: center

  Figure 2: The basic residual block with one neuron per hidden layer (source: [1])

Note that through this ResNet block there are two paths: one that goes through
a bottleneck of a *single* neuron, and one that is the identity function.  The
outputs then get added together at the end.  Writing out the expression of the
function, we have our ResNet block :math:`\mathcal{R}_i(x)` and our final network
:math:`Y(x)`:

.. math::

    \text{ReLU}(x) &= max(x, 0) \\
    \text{Id}(x) &= x \\
    \mathcal{R}_i({\bf x}) &= {\bf V_i}\text{ReLU}({\bf U_i}{\bf x} + b_i) + \text{Id}({\bf x}) \\
    Y(x) &= \mathcal{R}_n({\bf x}) \circ \ldots \circ \mathcal{R}_1({\bf x})
    \tag{1}

where :math:`U` is :math:`d x 1`, :math:`V` is :math:`1xd` matrix, and
:math:`b` is a constant.

Given the above ResNet architecture, the universal approximation theorem from
[1] states:

    ResNet with one single neuron per hidden layer is enough to provide
    universal approximation for any Lebesgue-integrable function as the depth
    goes to infinity.

This is a pretty surprising statement!  Remember, this architecture's only
non-linear element is a bottleneck with a *single* neuron!  However,
this does somehow imply the power of Residual Networks because even with a single
neuron it is powerful enough to represent any function.

|H2| Experiments |H2e|

In this section I played around with a bunch of different network architectures
to see how they would perform.  You can see my work in this notebook (TODO).

I generated three toy datasets labelled "easy", "medium", "hard" with two input
variables :math:`x_1, x_2`, and a single binary label.  Each dataset used the
same 300 :math:`(x_1, x_2)` input points but had a different predicate of
increasing complexity.  Figure 3 shows the three datasets.

.. figure:: /images/resnet_datasets.png
  :height: 200px
  :alt: "easy", "medium" and "hard" Datasets
  :align: center

  Figure 3: TODO TODO Plot of "easy", "medium" and "hard" Datasets.
  




|H3| The OG Universal Approximation Theorem |H3e|


|h2| Further Reading |h2e|

* Previous posts: `Residual Networks <link://slug/residual-networks>`__
* Wikipedia: `Universal Approximation Theorem <https://en.wikipedia.org/wiki/Universal_approximation_theorem>`__
* [1] `ResNet with one-neuron hidden layers is a Universal Approximator <https://arxiv.org/abs/1806.10909>`__, Hongzhou Lin, Stefanie Jegelka
* [2] `The Expressive Power of Neural Networks: A View from the Width <https://arxiv.org/abs/1709.02540>`__ Zhou Lu, Hongming Pu, Feicheng Wang, Zhiqiang Hu, Liwei Wang
