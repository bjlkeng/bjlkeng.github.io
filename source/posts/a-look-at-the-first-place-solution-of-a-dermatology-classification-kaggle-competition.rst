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

MobileNetV2 [2_] introduced a new type of neural network architectural building
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
hypothesis the [2_] proposes is that ReLU non-linearity on the inverted
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
(Note: Not shown in the above code is that `BatchNormalization <https://en.wikipedia.org/wiki/Batch_normalization>`__
is applied after every convolution layer (but before the activation).)

The resulting MobileNetV2 architecture is very memory efficient for mobile
applications as the name suggests.  Generally, the paper shows that MobileNetV2 
uses less memory and computation with similar (sometimes better) performance
on standard benchmarks.  Details on the architecture can be found in [2_].

Squeeze and Excitation Optimization
-----------------------------------

The Squeeze and Excitation (SE) block [3_] is an optimization that can added on to a
convolutional layer that scales each channel's outputs by using a learned
function of the average activation of each channel.  The basic idea is shown in
Figure 1 where from a convolution operation (:math:`F_tr`), we branch off to
calculate a scalar per channel ("squeeze" via :math:`F_sq`), pass it through some layers
("excite" via :math:`F_ex`), and then scale the original convolutional outputs using the SE block.
This can be thought of as a self-attention mechanism on the channels.

.. figure:: /images/dermnet_squeeze_excite.png
  :height: 200px
  :alt: Squeeze Excite
  :align: center

  **Figure 1: Squeeze Excitation Block with ratio=1 [** 3_ **]**

The main problem the SE block addresses is that each convolutional output pixel only
looks at it's local receptive field (e.g. 3x3).  A convolutional network only
really considers global spatial information by stacking multiple layers, which
seems inefficient.  Instead, the hypothesis of the SE block is that you can model
the global interdependencies between channels and allow each channel to
increase their sensitivity improving learning.

Code for an SE block is shown in  Listing 4.  First, we do a
:code:`GlobalAveragePool2D`, which amounts to compute the mean for each
channel.  Then we pass it through two 1x1 convolutional layers with a ReLU and
sigmoid activation respectively.  The first convolutional layer can be thought
of as "mixing" the averages across the channel, while the second one converts
it to a value between 0 and 1.  It's not clear whether more or less layers is better
but [3_] says that they wanted to limit the added model complexity while still
having some generalization power.

.. code-block:: Python

    def squeeze_excite(x, filters, ratio=4):
        # computes mean of each spatial dimensions (outputs a mean value for each channel)
        m1 = GlobalAveragePooling2D(keepdims=True)(x) 
        m2 = Conv2D(filters // ratio, (1, 1), activation='relu')(m1)
        m3 = Conv2D(filters, (1, 1), activation='sigmoid')(m2)
        return Multiply(m3, x)

**Listing 4: SqueezeExcite block in Keras** (adapted from `source <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py#L103>`__)

Since the SE block only operates on the channels due to the :code:`GlobalAveragePool2D` so
the added computational and memory requirements are modest.  The largest contributors are
usually the latter layers that have a lot of channels.  In their experiments,
the parameters of a MobileNet network increased by roughly 12% but was able to improve
the ImageNet top-1 error rate by about 3% [3_].  Overall, it seems like a nice little
optimization that improves performance across a wide variety of visual tasks.


EfficientNet
------------

EfficientNet is a convolutional neural networks (ConvNet) architecture [4_]
(circa 2019) that rethinks the standard ConvNet architecture choices and
proposes a new architecture family called *EfficientNets*.  The first main idea
is that ConvNets can be scaled to have more capacity in three broad network dimensions
shown in Figure 2:

* **Wider**: In the context of ConvNets, this corresponds to more channels per layer (vs. more neurons in a fully connected layer).
* **Deeper**: Deeper means more convolutional layers.
* **Higher Resolution**: Means using higher resolution inputs (e.g. 560x560 vs. 224x224 images).

.. figure:: /images/dermnet_scaling.png
  :height: 470px
  :alt: Scaling ConvNets
  :align: center

  **Figure 2: Model scaling figure from [** 4_ **]: (a) base model, (b) increase width, (c) increase depth, (d) increase resolution.**

The first insight [4_] found is that, as expected, scaling the
above network dimensions result in better ConvNet accuracy (as measured via top-1
ImageNet accuracy) but with diminishing returns.  To standardize the evaluation,
they normalize the scaling using FLOPS.

The next logical insight discussed in [4_] is that balancing
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

A specific EfficientNet architecture is also proposed in [4_] that defines
a base architecture labeled "B0" shown in Figure 3 using the above MBConv
MobileNetV2 block discussed above with the Squeeze and Excitation optimization
added to each block.  Overall the base B0 architecture is a typical ConvNet
where in each layer the resolution decreases but channels increase.

.. figure:: /images/dermnet_effnet.png
  :height: 270px
  :alt: Effnet architecture
  :align: center

  **Figure 3: EfficientNet-B0 baseline archiecture [** 4_ **]**

From the B0 architecture, we can derive scaled architectures labeled
B1-B7 by:

1. Fix :math:`\phi=1` and assume two times more resources are available (see Equation 1),
   and do a small grid search to find :math:`\alpha, \beta, \gamma`, which were
   :math:`\alpha=1.2, \beta=1.1, \gamma=1.15` (depth, width, resolution, respectively),
   which give roughly 1.92 according to Equation 1.
2. Scale up the B0 architecture approximately using Equation 1 with the
   constants described in Step 1 by increasing :math:`\phi` (and round where
   appropriate).  Dropout is increased roughly linearly as the architectures
   grow from B0 (0.2) to B7 (0.5).

Table 1 shows the flops, multipliers and dropout rate for each dimension.

.. csv-table:: Table 1: EfficientNet Architecture Multipliers (`source <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/gen_efficientnet.py#L502>`__)
   :header: "Name","FLOPs","Depth Mult.","Width Multi.","Resolution","Dropout Rate"
   :widths: 8,5,5,5,5,5
   :align: center

    efficientnet-b0,0.39B,1.0,1.0,224,0.2
    efficientnet-b1,0.70B,1.1,1.0,240,0.2
    efficientnet-b2,1.0B,1.2,1.1,260,0.3
    efficientnet-b3,1.8B,1.4,1.2,300,0.3
    efficientnet-b4,4.2B,1.8,1.4,380,0.4
    efficientnet-b5,9.9B,2.2,1.6,456,0.4
    efficientnet-b6,19B,2.6,1.8,528,0.5
    efficientnet-b7,47B,3.1,2.0,600,0.5

..
    Depth Mult.	Width Multi.	Resolution
    1.00	1.00	224.00
    0.52	0.00	0.49
    1.00	1.00	1.07
    1.85	1.91	2.09
    3.22	3.53	3.78
    4.32	4.93	5.09
    5.24	6.17	6.14
    6.21	7.27	7.05

For example, starting with B0, we have 0.39B FLOPs, going to B4 we have 4.2B
flops, which yields :math:`\phi = 4.2 / 0.39 \approx 3.28`.  This translates to
scaling close to this value along the three dimensions with :math:`\phi_{\alpha} = 3.22`,
:math:`\phi_{\beta}=3.53`, and :math:`\phi_{\gamma}=3.78`.  We're not going for
precision here, we just want a rough guideline of how to scale up the
architecture.  The nice thing about having this guideline is that we can create
bigger ConvNets without having to do any additional architecture
search.


Noisy Student
-------------

Noisy Student [5_] is a semi-supervised approach to training a model that is
useful even when you have abundant lableled data.  This work is in the context
of images where they show its efficacy on ImageNet and related benchmarks.
The setup requires both labelled data and unlabeled data with a relatively
simple algorithm (with some subtlety) and the following steps:

1. Train teacher model :math:`M^t` with labelled images using a standard cross
   entropy loss.
2. Use the :math:`M^t` (current teacher) to generate pseudo labels for the unlabelled data
   (**filter and balance dataset as required**)
3. Learn a student model :math:`M^(t+1)` with **equal or larger** capacity
   on the labeled and unlabeled data with added **noise**.
4. Increment :math:`t` (make the current student the new teacher) and **repeat**
   steps 2-3 as needed.

A few unintuitive points emphasized in bold.  First, the student model uses a
equal or larger model.  This is different from other student/teacher context
where one is trying to distill the model knowledge into the smaller model.
Here we're not trying to distill, we're trying to boost performance so we want
a bigger model so it can learn from the bigger combined dataset.  This seems to
have a increase of 0.5-1.5% in top-1 ImageNet accuracy in their ablation
study.

Second, the noise is implemented as randomized data augmentation plus dropout
and stochastic depth.  The added noise on the student seems to around another 0.5%
in top-1 ImageNet accuracy.  Seems like a reasonable modification given that
you typically want both of these things when training these types of networks.

Third, the iteration in step 4 also seemed important.  Going from one iteration
to 3 improved performance by 0.8% in top-1 ImageNet accuracy.  It's not obvious
to me that the performance would improve by iterating here but since the number
of iterations is small, I can believe that it's possible.

Lastly, they discuss that they filter out pseudo labels that have low
confidence by the teacher model, and then rebalance the unlabelled classes so
the distribution is not so off (by repeating images).  This also seems to
improve performance a bit more modestly at 0-0.3% depending on the model.

The summary of the overall Noisy Student results are shown in Figure 4 where
they conducted most of their experiments on EfficientNet.  This figure only
shows the non-iterative training (their headline result is within the iterative
training).  You can see that the Noisy Student dominates the vanilla
EfficientNet results at the same number of model parameters and achieves SOTA
(at the time of the paper).  In the context of this post, there are many
versions of EfficientNet with Noisy Student training that are available to use
a pretrained model.

.. figure:: /images/dermnet_noisystudent.png
  :height: 470px
  :alt: Noisy Student
  :align: center

  **Figure 4: Noisy Student Training shows significant improvement over all model sizes. [** 5_ **]**


SIIM-ISIC Melanoma Classification 2020 Competition
==================================================

The Society for Imaging and Informatics in Medicine (SIIM) and the International Skin Imaging Collaboration (ISIC)
melanoma classification competition [0_] aims to classify a given skin lesion
as melanoma along with accompanying patient metadata.  Melanoma is a type of
skin cancer that is responsible for over 75% of skin cancer deaths.  The ISIC
has been putting on various computer vision `challenges <https://challenge.isic-archive.com/>`__ related to dermatology since 2016.
Notably, past competitions have labelled image skin lesion data (and sometimes
patient metadata) but with different labels that may be a superset of the 2020 competition.
More than 3300 teams participated in the competition with the winning solution
being the topic of this post [1_]. 

The dataset consists of 33k training data points with only 1.76% positive samples (i.e., melanoma).
Each datum contains a 1024x1024 image of a skin lesion along with patient data: 

* patient id
* sex
* approximate age
* location of image site
* detailed diagnosis (training only)
* benign or malignant (training only, label to predict)
* binarized version of target

The competition in 2020 was hosted on Kaggle which contained a leaderboard of
all submissions.  Each team submitted a blind prediction on the given test set
and the leaderboard will measure its performance using AUC.
The leaderboard will show a public view on all submissions which shows the AUC
score based on 30% of the test set.  The remaining 70% will be hidden on the
private leaderboard until the end of the competition and be used to evaluate
the final result.

Table 2 shows several select submissions including the top 3 on the public and
private leaderboards.  Interestingly, the top 3 winners on the private data all
ranked relatively low, including the top submission which ranked all the way
down at 881!  Impressively, the top public score had a whopping 0.9931 AUC but
only ended up at rank 275 in the final private ranking.  The number of submissions
is also interesting.  Clearly, some overfitting on this test set was going on
in certain submissions with the top 3 winners all having relatively low number
of submissions compared to others.  The other obvious thing is that 
the scores are so close together that luck definitely played a role in the
submissions.

.. csv-table:: Table 2: Performance of Select Teams (`source <https://www.kaggle.com/competitions/siim-isic-melanoma-classification/leaderboard>`__)
    :header: Private Rank,Private Score,Public Rank,Public Score,Submissions  
    :widths: 4,3,4,3,4
    :align: center

    1,0.9490,881,0.9586,116
    2,0.9485,57,0.9679,61
    3,0.9484,265,0.9654,118
    27,0.9441,2,0.9926,402
    100,0.9414,329,0.9648,121
    275,0.9379,1,0.9931,276
    395,0.9357,3,0.9767,245
    500,0.9336,241,0.9656,227


Architecture
============


Problem Formulation
===================

Implementation
==============


Experiments
===========

* Experiment with just 2020 data
* Experiment with just binarized labels
* Experiment with/without patient data


Discussion and Other Topics
===========================


Conclusion
==========


Further Reading
===============


.. _0: 

[0] `SIIM-ISIC Melanoma Classification Kaggle Competition <https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview>`__

.. _1: 

[1] Qishen Ha, Bo Liu, Fuxu Liu, "Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge", `<https://arxiv.org/abs/2010.05351>`__

.. _2:

[2] Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018, `<https://arxiv.org/abs/1801.04381>`__

.. _3:

[3] Hu et al. "Squeeze-and-Excitation Networks", CVPR 2018, `<https://arxiv.org/abs/1801.04381>`__

.. _4:

[4] Mingxing Tan, Quoc V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", `<https://arxiv.org/abs/1905.11946>`__

.. _5:

[5] Xie et al. "Self-training with Noisy Student improves ImageNet classification", `<https://arxiv.org/abs/1911.04252>`__
