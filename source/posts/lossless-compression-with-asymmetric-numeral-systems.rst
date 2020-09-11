.. title: Lossless Compression with Asymmetric Numeral Systems
.. slug: lossless-compression-with-asymmetric-numeral-systems
.. date: 2020-09-01 18:37:43 UTC-04:00
.. tags: compression, entropy, asymmetric numeral systems, Huffman coding, Arithmetic Coding, mathjax
.. category: 
.. link: 
.. description: A post on Asymmetric Numeral Systems coding
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

During my undergraduate, one of the most interesting courses I took was on
coding and compression.  Here was a course that combined algorithms,
probability and secret messages, what's not to like?  (I actually like most of
the communication theory courses, it was interesting to learn the basics of how
wired/wireless signals are transmitted)  I ended up not going down this career
path, at least partially because communications systems had its heyday around
the 2000s with companies like Nortel and Blackberry (and its predecessors).
Some like to joke that all the major theoretical breakthroughs were done by
Shannon with his discovery of information theory.
Fortunately, I eventually wound up studying industrial applications of
classical AI techniques and then machine learning, which has really grown like
crazy in the last 10 years or so.  Which is exactly why I was so surprised that
a *new* and *better* method of lossless compression was developed in 2009
*after* I finished my undergraduate when I was well into my PhD.  It's a bit
mind boggling that something as well-studied as entropy-based lossless
compression still had (have?) totally new methods to discover, but I digress.

In this post, I'm going to write about a relatively new entropy based encoding
method called Asymmetrical Numeral Systems (ANS) developed by Jaroslaw (Jarek)
Duda [2].  If you've ever heard of Arithmetic Coding (probably best known for
its use in JPEG compression), ANS runs in a very similar vein.  It can
theoretically codes that are close to the theoretical compression limit
(similar to Arithmetic coding) but is *much* faster.  It's been used in 
modern compression algorithms since 2014 including compressors developed
by Facebook, Apple and Google [3].  As usual, I'm going to go over some
background, some intuition backed by math, examples to help with intuition and
finally some experiments I ran with a toy ANS implementation I wrote.  I
hope you're as excited as I am, let's begin!

.. TEASER_END

|h2| Background: Data Coding and Compression |h2e|

Often you will want to **code** information (such as a file) for storage or
transmission on communication channel by specifying some rules on how to
transform the information.  The usual reasons why you want to code a message is
compression (fewer bits), redundancy (error correction/detection), or
encryption (confidentiality).  For compression, we generally have two main
approaches: lossy and lossless.  In lossy schemes, you drop some non-essential
details in the message (think image compression) to trade-off a greater
reduction in size.  This trick is used extensively in things like image 
(e.g. JPEG) or audio (e.g. MP3) compression.  Lossless schemes on the other
hand aim to retain the exact message reducing the file size by exploiting some
statistical redundancies in the data.  We're mainly talking about lossless
schemes today.

Lossless schemes come in many forms such as
`Run Length Encoding <https://en.wikipedia.org/wiki/Run-length_encoding>`__,
`Lempel-Ziv compression <https://en.wikipedia.org/wiki/LZ77_and_LZ78>`__,
`Huffman coding <https://en.wikipedia.org/wiki/Huffman_coding>`__, and
`Arithmetic coding <https://en.wikipedia.org/wiki/Arithmetic_coding>`__,
which probably constitute the most popular ones (aside from ANS, which we'll be
discussing today).  Most of the ones above work by reading one or more symbols
(think bytes) from the data stream and replacing it with some compact bit
representation.  For example, in Huffman coding, the most frequent symbol is
replaced with a single bit, or Run Length Encoding, which replaces a repeated
sequence of a character by a character and how many times it repeats.  In both
these examples, it works on a subset of the sequence and replaces them.  The
other variant which both Arithmetic Coding and Asymmetrical Numeral Systems
fall under is where then *entire* message is encoded as a single number (the
smaller or less precise the number, the shorter the message).  This allows you
to get closer to the theoretical compression limit.

Speaking of theoretical compression limits, according to Shannon's
`source coding theorem <https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem>`__
the theoretical limit you can (losslessly) compress data is equal 
to its entropy.  In other words, the average number of bits per symbol
cannot be smaller than:

.. math::

    H(X) = -\sum_{i=1}^n p_i \log_2 p_i  \tag{1}

where it's presumed that you know the :math:`p_i` distribution of each of your :math:`n` symbols 
ahead of time.  I wrote some details on how to think about entropy in my previous post on
`maximum entropy distributions <link://slug/maximum-entropy-distributions>`__ so I won't
go into much detail now.



a class of lossless compression
encoders are called `entropy encoders <https://en.wikipedia.org/wiki/Entropy_encoding>`__.


* Lossy ones
* Entropy Coders
    * Huffman
    * Arithmetic
    * Ranged 

.. math::

    a = 1

|h2| Asymmetric Numeral Systems |h2e|

|h3| Concept |h3e|

|h3| Uniform Binary Variant (uABS) |h3e|

|h3| Range variants (rANS) |h3e|

* rANS
* Renormalization

|h3| Other variants |h3e|

* tANS


|h2| Experiments |h2e|

|h2| References |h2e|

* [1] "Lecture I: data compression ... data encoding", Jaroslaw Duda, Nokia Krakow, `<http://ww2.ii.uj.edu.pl/~smieja/teaching/ti/3a.pdf>`__ 
* [2] "Asymmetric numeral systems", Jarek Duda, `<https://arxiv.org/abs/0902.0271>`__
* [3] Wikipedia: `<https://en.wikipedia.org/wiki/Asymmetric_numeral_systems`__
