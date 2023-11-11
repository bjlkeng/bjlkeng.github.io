.. title: A Look at The First Place Solution of a Dermatology Classification Kaggle Competition
.. slug: a-look-at-the-first-place-solution-of-a-dermatology-classification-kaggle-competition
.. date: 2023-11-11 13:09:46 UTC-05:00
.. tags: dermatology, effnet, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

One interesting I often wonder about is the gap between academic solutions and
practical real-world solutions.  In general academic solutions are confined to
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
<https://arxiv.org/abs/2010.05351>`__ [1] to the 
`SIIM-ISIC Melanoma Classification <https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview>`.
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


* `SIIM-ISIC Melanoma Classification Kaggle Competition <https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard>`__
* [1] Qishen Ha, Bo Liu, Fuxu Liu, "Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge", `<https://arxiv.org/abs/2010.05351>`__
