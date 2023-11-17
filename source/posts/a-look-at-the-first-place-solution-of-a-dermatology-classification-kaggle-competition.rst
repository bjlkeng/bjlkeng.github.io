.. title: A Look at The First Place Solution of a Dermatology Classification Kaggle Competition
.. slug: a-look-at-the-first-place-solution-of-a-dermatology-classification-kaggle-competition
.. date: 2023-11-11 13:09:46 UTC-05:00
.. tags: dermatology, effnet, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

One interesting I often think about is the gap between academic and real-world
solutions.  In general academic solutions are confined to
a narrow problem space, more often than not needing to care about the
underlying data (or its semantics).  `Kaggle <https://www.kaggle.com/competitions>`__
competitions are a (small) step in the right direction usually providing a true
blind test set and opening a few degrees of freedom in terms of using any
technique that will have an appreciable effect on the performance, which
usually eschews novelty in favour of more robust methods.  To this end, I
thought it would be useful to take a look at a more realistic scenario (via a
Kaggle competition) and understand the practical details that gets you superior
performance on a more realistic task.

This post will cover the `first place solution
<https://arxiv.org/abs/2010.05351>`__ [1_] to the 
`SIIM-ISIC Melanoma Classification <https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview>`_ [0_].
In addition to using tried and true architectures (mostly EfficientNets), they
have some interesting tactics they use to formulate the problem, process the
data, and train/validate the model.  I'll provide the background on the
competition/data, architectural details, problem formulation, and
implementation details.  I've also run some experiments to better understand
the benefit of certain architectural decision they made.  Enjoy!


.. TEASER_END
.. section-numbering::
.. raw:: html

    <div class="card card-body bg-light">
    <h1>Table of Contents</h1>

.. contents:: 
    :depth: 2
    :local:

.. raw:: html

    </div>
    <p>


Background
==========

Inverted Residuals and Linear Bottlenecks (MobileNetV2)
-------------------------------------------------------

MobileNetV2 [3_] introduced a new type of neural network architectural building
block often known as "MBConv".  The two big innovations here are inverted residuals
and linear bottlenecks.  

First to understand inverted residuals, let's take a look at the basic
residual block (also see my post on `ResNet <link://slug/residual-networks>`__)
shown in Listing 1.  Two notable parts of the basic residual block are the
fact that we reduce the number of channels to :code:`squeeze` and then grow the
number of channels with :code:`expand`.  The squeeze operation is often known
as a *bottleneck* since we have fewer channels.  The intuition here is to reduce
the number of channels so that the more expensive 3x3 convolution is cheaper.
The other relevant part is that fact that we have the residual "skip" connection where
we add the input to the result of the transformations.  Notice the residual
connection connects the expanded parts :code:`x` and :code:`m3`.

.. code-block:: Python

   def residual_block(x, squeeze=16, expand=64):
       # x has 64 channels in this example
       m1 = Conv2D(squeeze, (1,1), activation='relu')(x)
       m2 = Conv2D(squeeze, (3,3), activation='relu')(m1)
       m3 = Conv2D(expand, (1,1), activation='relu')(m2)
       return Add()([m3, x])

**Listing 1: Example of a Basic Residual Block in Keras** (adapted from `source <https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5>`__)

Next, let's look at what changes with an inverted residual shown in Listing 2.
Here we "invert" the residual connection where we are making the residual
connection between the bottleneck "squeezed" layers.  Recall that we'll
eventually be stacking these blocks, so there will still be alternations
of squeezed ("bottlenecks") and expansion layers.  The difference with
Listing 1 is that we'll be making residual connections between the bottleneck
layers instead of expansion layers.  

.. code-block:: Python

   def inverted_residual_block(x, expand=64, squeeze=16):
       # x has 16 channels in this example
       m1 = Conv2D(expand, (1,1), activation='relu')(x)
       m2 = DepthwiseConv2D((3,3), activation='relu')(m1)
       m3 = Conv2D(squeeze, (1,1), activation='relu')(m2)
       return Add()([m3, x])

**Listing 2: Example of an inverted residual block with depthwise convolution in Keras** (adapted from `source <https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5>`__)


The other thing to note is that the 3x3
convolution is now expensive if we do it on the expanded layer so instead we'll 
use a `depthwise convolution <https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/>`__
for efficiency.  This reduces reduces the number of parameters needed from
:math:`h\cdot w \cdot d_i \cdot d_j \cdot k^2` for a regular 3x3 convolution to
:math:`h\cdot w \cdot d_i (k^2 + d_j)` for a depthwise convolution where
:math:`h, w` are height and width, :math:`d_i, d_j` are input/output channels, and
:math:`k` is the convolutional kernel size.  With :math:`k=3` this could potentially
reduce the number of parameters needed by 8-9 times with only a small hit to
accuracy.

.. code-block:: Python

   def inverted_linear_residual_block(x, expand=64, squeeze=16):
       m1 = Conv2D(expand, (1,1), activation='relu')(x)
       m2 = DepthwiseConv2D((3,3),  activation='relu')(m1)
       m3 = Conv2D(squeeze, (1,1))(m2)
       return Add()([m3, x])

**Listing 3: MBConv Block in Keras** (adapted from `source <https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5>`__)

The last big thing thing that MBConv block changed was removing the
non-linearity on bottleneck layer as shown in Listing 3.  A
hypothesis the [3_] proposes is that ReLU non-linearity on the inverted
bottleneck hurts performance.  The idea is that ReLU either is the identify
function if the input is positive, or zero otherwise.  In the case that the
activation is positive, then it's simply a linear output so removing the
non-linearity isn't a bit deal.  On the other hand, if the activation is
negative then ReLU actively discards information (e.g., zeroes the output).
Generally for wide networks (i.e., lots of convolutional channels), this is not
a problem because we can make up for information loss in the other channels.
In the case of our squeezed bottleneck though, we have fewer layers so we lose
a lot more information, hence hurt performance.  The authors note that this
effect is lessened with skip connections but still present.

The resulting MobileNetV2 architecture is very memory efficient for mobile
applications as the name suggests.  Generally, the paper shows that MobileNetV2 
uses less memory and computation with similar (sometimes better) performance
on standard benchmarks.  Details on the architecture can be found in [3_].

Squeeze and Excitation Optimization
-----------------------------------
[4_]


EfficientNet
------------

EfficientNet is a convolutional neural networks (ConvNet) architecture [2_]
(circa 2019) that rethinks the standard ConvNet architecture choices and
proposes a new architecture family called *EfficientNets*.  The first main idea
is that ConvNets can be scaled to have more capacity in three broad network dimensions
shown in Figure 1:

* **Wider**: In the context of ConvNets, this corresponds to more channels per layer (vs. more neurons in a fully connected layer).
* **Deeper**: Deeper means more convolutional layers.
* **Higher Resolution**: Means using higher resolution inputs (e.g. 560x560 vs. 224x224 images).

.. figure:: /images/dermnet_scaling.png
  :height: 470px
  :alt: Scaling ConveNet
  :align: center

  **Figure 1: Model scaling figure from [** 2_ **]: (a) base model, (b) increase width, (c) increase depth, (d) increase resolution.**

The first insight [2_] found is that, as expected, scaling the
above network dimensions result in better ConvNet accuracy (as measured via Top-1
ImageNet accuracy) but with diminishing returns.  To standardize the evaluation,
they normalize the scaling using FLOPS.

The next logical insight discussed in [2_] is that balancing
how all three scaling network dimensions is important to 
efficiently scale ConveNets.  They propose a compound
scaling method as:

.. math::

    \text{depth}: d &= \alpha^\phi \\
    \text{width}: w &= \beta^\phi \\
    \text{resolution}: r &= \gamma^\phi \\
        \text{s.t. }\hspace{10pt} \alpha&\cdot\beta^2\cdot\gamma^2 \approx 2 \\
    \alpha \geq 1, \beta &\geq 1, \gamma \geq 1 \\
    \tag{1}

The intuition here is that we want to be able to scale the network
size appropriately for a given FLOP budget, and Equation 1, if satisfied, will
approximately scale the network by :math:`(\alpha \cdot \beta^2 \cdot \gamma^2)^\phi`.
Thus, :math:`\phi` is our user-specified scaling parameter while
:math:`\alpha, \beta, \gamma` are how we distribute the FLOPs to each scaling
dimension and are found by a small grid search.  The constraint 
:math:`\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2` (I believe) is arbitrary
so that the FLOPS will increase by roughly :math:`2^\phi`.  Additionally,
it likely simplifies the grid search that we need to do.




SIIM-ISIC Melanoma Classification
=================================

Data
----

Architecture
============


Problem Formulation
===================

Implementation
==============


Experiments
===========


Discussion and Other Topics
===========================


Conclusion
==========


Further Reading
===============


.. _0: 

[0] `SIIM-ISIC Melanoma Classification Kaggle Competition <https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard>`__

.. _1: 

[1] Qishen Ha, Bo Liu, Fuxu Liu, "Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge", `<https://arxiv.org/abs/2010.05351>`__

.. _2:

[2] Mingxing Tan, Quoc V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", `<https://arxiv.org/abs/1905.11946`>__

.. _3:

[3] Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018, `<https://arxiv.org/abs/1801.04381>`__

.. _4:

[4] Hu et al. "Squeeze-and-Excitation Networks", CVPR 2018, `<https://arxiv.org/abs/1801.04381>`__
