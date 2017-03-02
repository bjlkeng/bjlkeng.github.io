.. title: Variational Bayes and The Mean-Field Approximation
.. slug: variational-bayes-and-the-mean-field-approximation
.. date: 2017-03-02 08:02:46 UTC-05:00
.. tags: Bayesian, variational calculus, mean-field, Kullback-Leibeler 
.. category: 
.. link: 
.. description: A brief introduction to variational Bayes and the mean-field approximation.
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

This post is going to cover Variational Bayesian methods and, in particular,
the most common one, the mean-field approximation.  This is a topic that I've
been trying to get to for a while now but didn't quite have all the background
that I needed.  After picking up the main ideas from
`Variational Calculus <link://slug/the-calculus-of-variations>`__ and
getting more fluent in manipulating probability statements like
in my `EM <link://slug/the-expectation-maximization-algorithm>`__ post,
this variational Bayes stuff seems a lot easier.

Variational Bayesian methods are a set of techniques to approximate posterior
distributions in `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__.
If this sounds a bit terse, keep reading!  I hope to provide a bunch of intuition
so that the big ideas are easy to understand (which they are), but of course we 
can't do that well unless we have a healthy dose of mathematics.  For some of the
background concepts, I'll try to refer you to good sources (including my own),
which I find is the main blocker to understanding this subject.  Enjoy!

.. TEASER_END

|h2| Kullback-Leibeler Divergence and Finding Like Probability Distributions |h2e|



|h2| Further Reading |h2e|

* Previous Posts: `Variational Calculus <link://slug/the-calculus-of-variations>`__, `Expectation-Maximization Algorithm <link://slug/the-expectation-maximization-algorithm>`__, `Normal Approximation to the Posterior <link://slug/the-expectation-maximization-algorithm>`__
* Wikipedia: `Variational Bayesian methods <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`__, `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`__
