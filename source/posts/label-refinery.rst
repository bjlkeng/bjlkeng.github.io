.. title: Label Refinery
.. slug: label-refinery
.. date: 2018-08-27 08:26:02 UTC-04:00
.. tags: label refinery, residual networks, CIFAR10, svhn, mathjax
.. category: 
.. link: 
.. description: A short post on the "Label Refinery" paper.
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

This post is going to be about a really simple idea that is surprisingly effective
from a paper by Bagherinezhad et al. called `Label Refinery: Improving ImageNet
Classification through Label Progression <https://arxiv.org/abs/1805.02641>`__.
The title pretty much says it all but I'll also discuss some intuition and some
experiments on the CIFAR10 and SVHN datasets.  The idea is both simple and
unintuitive, my favourite kind of idea!  Let's take a look.

.. TEASER_END

|H2| Label Refinery |H2e|

|H3| Main Idea |H3e|

Let's start off with the main idea because it's so simple.  All we're 
going to do is train an image classification model, use its predicted outputs
:math:`y'` to train a *fresh* classification models using :math:`y'` in place
of the ground truth labels.  That's right: we're using the predicted labels
in place of ground truth labels!  The claim is that this "refinement" step
will improve your out of sample accuracy.  Figure 1 from the paper shows this
visually.

.. figure:: /images/label_refinery1.png
  :height: 200px
  :alt: Label Refinery Illustration
  :align: center

  Figure 1: The basic idea behind Label Refinery (source: [1])

You can see at each iteration we're using the predicted labels ("Refined
label") to feed into the next model as the training labels.  At point, the last
model in the chain becomes your classifier.  Along the way, inexplicably, the
accuracy improves.

In more detail, for a given image classification problem:

0. Set :math:`y = y_{\text{truth}}` (ground truth labels).
1. Train a classifier :math:`R_1` with images :math:`X` and labels :math:`y`
   from scratch.
2. Use :math:`R_1` to predict on :math:`X` to generate new "refined" labels
   :math:`y=y_{R_1}`.
3. Repeat steps 1-2 several times to iteratively generate new models
   :math:`R_i` and "refined" labels :math:`y_{R_i}`.
4. At some point, use `R_i` as your trained model.

But why should it even work?  How can using a *less* accurate label (predicted
output) to train result in better accuracy?  It's actually quite simple.

|H3| Intuition |H3e|

The big idea is that *images can have multiple labels*! Well... duh!  However,
this is a big problem for datasets like ImageNet where each image has exactly
one label.  Take Figure 2 for example.

.. figure:: /images/label_refinery2.png
  :height: 200px
  :alt: Label Refinery Illustration
  :align: center

  Figure 2: Illustration of multiple label problem (source: [1])

Figure 2 shows a picture of a "persian cat" from imageNet's training set.  This
label is not bad but you could also conceivably label this image as a "ball" too.
This problem is magnified though when training a network using standard image augmentation
techniques such as cropping.  We can see the cropped image should clearly be
labelled as a "ball" and not "persian cat".  However, if we add this augmented cropped
image to our dataset, it will just add noise making it harder for the
classifier to learn.  On the other hand, if we use the label refinery technique above,
we can have a *soft labelling* of the image.  So it could be "80% persian cat" and
"20% ball".  This help reduce overfitting to the training images.  
The next figure shows another example of this.

.. figure:: /images/label_refinery3.png
  :height: 200px
  :alt: Label Refinery Illustration
  :align: center

  Figure 3: Examples of similar image patches. (source: [1])

Figure 3 shows examples of random crops of "dough" and "butternut squash".  You can
see that they are visually very similar.  Similarly, if we use hard labels for the
crops, we'll most likely have a very dissonant time learning because for two
similar images we have completely different labels.  Contrast that with having a 
"soft label" for the patches of "dought", "buttnut squash", "burrito", "french
loaf" etc.

.. figure:: /images/label_refinery4.png
  :height: 400px
  :alt: Label Refinery Illustration
  :align: center

  Figure 4: Examples of similar image patches. (source: [1])

Figure 4 shows another example with the top three predictions at each iteration
of label refinery.  You can see that the first iteration clearly overfit to
"barbershop" especially when considering the cropped "shop" label.  Using
label refinery on the fifth iteration, we have a more reasonable soft labelling
of "barbershop", "scoreboard", and "street sign".


|H2| Experiments on CIFAR10 and SVHN |H2e|

|H2| Conclusion |H2e|

|H2| Further Reading |H2e|

* Previous posts: `Residual Networks <link://slug/residual-networks>`__
* [1] `Label Refinery: Improving ImageNet Classification through Label Progression <https://arxiv.org/abs/1805.02641>`__, Hessam Bagherinezhad, Maxwell Horton, Mohammad Rastegari, Ali Farhadi
