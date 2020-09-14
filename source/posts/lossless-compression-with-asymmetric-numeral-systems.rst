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

Most modern digital communication is done via transmitting **bits** 
(i.e. 0/1 values) with the goals of both reliability and efficiency.
A **symbol** is unit of information that is transmitted, which can be the bits itself
or a higher level concept represented by a sequence of bits (e.g. "a", "b", "c", "d" etc.).
An **alphabet** is the set of all symbols that you can transmit.
A **message** is a sequence of symbols that you want to transmit.

Often you will want to **code** a message (such as a file) for storage or
transmission on communication channel by specifying some rules on how to
transform it into another form.  The usual reasons why you want to code a message is
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
representation.  For example, in Huffman Coding, the most frequent symbol is
replaced with a single bit, or Run Length Encoding, which replaces a repeated
sequence of a character by a character and how many times it repeats.  In both
these examples, it works on a subset of the sequence and replaces them.  The
other variant which both Arithmetic coding and Asymmetrical Numeral Systems
fall under is where then *entire* message is encoded as a single number (the
smaller or less precise the number, the shorter the message).  This allows you
to get closer to the theoretical compression limit.

Speaking of theoretical compression limits, according to Shannon's
`source coding theorem <https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem>`__
the theoretical limit you can (losslessly) compress data is equal 
to its entropy.  In other words, the average number of bits per symbol of your
message cannot be smaller than:

.. math::

    H(X) = -\sum_{i=1}^n p_i \log_2(p_i)  \tag{1}

where it's presumed that you know the :math:`p_i` distribution of each of your :math:`n` symbols 
ahead of time.  Note that the logarithm is base 2, which naturally allows us to
talk in terms of "bits".
I wrote some details on how to think about entropy in my previous post on
`maximum entropy distributions <link://slug/maximum-entropy-distributions>`__ so I won't
go into much detail now. 

.. admonition:: Example 1: Entropy of a Discrete Probability Distribution

    Imagine we have an alphabet with 3 symbols: :math:`\{a, b, c\}`.  
    Let random variable :math:`X` represent the probability of seeing a symbol
    in a message, and is given by the distribution: 
    :math:`p_a = \frac{4}{7}, p_b=\frac{2}{7}, p_c=\frac{1}{7}`.
    
    The entropy and minimum average number of bits per symbol we can achieve
    for messages with this distribution (according the the source coding
    theorm) is:

    .. math::

         H(X) &= -\sum_{i=1}^n p_i \log_2(p_i)  \\
              &= -p_a\log_2(p_a)  - p_b\log_2(p_b) - p_c\log_2(p_c) \\
              &= -\frac{4}{7}\log_2(\frac{4}{7})
                 - \frac{2}{7}\log_2(\frac{2}{7})
                 - \frac{1}{7}\log_2(\frac{1}{7}) \\
              &\approx 1.3788 bits \\
              \tag{2}
    
    Contrast that to naively encoding each symbol using 2-bits (vs. 1.3788),
    for example, representing "a" as 00, "b" as 01, and "c" as 10 (leaving 11
    unassigned).  
    
    This can also be contrasted to assuming that we had a uniform distribution
    (:math:`p_a=p_b=p_c=\frac{1}{3}`), which would yield us an entropy of
    :math:`H(X)=-3\cdot\frac{1}{3}log_2(\frac{1}{3}) = 1.5850` vs. 1.3788 with
    a more skewed distribution.  This shows a larger idea that uniform
    distributions are the "hardest" to compress (i.e. have the highest entropy)
    because you can't really exploit any asymmetry in the symbol distribution
    -- all of them are equally likely.
 

One class of lossless compression schemes is called
`entropy encoders <https://en.wikipedia.org/wiki/Entropy_encoding>`__ and they
exploit the estimated statistical properties of your message in order to get
pretty close to the theoretical compression limit.  Huffman coding, Arithmetic
coding, and Asymmetric Numeral Systems all are entropy encoders.

Finally, the metric we'll be using is compression ratio, defined as:

.. math::

    \text{Compression Ratio} = \frac{Uncompressed Size}{Compressed Size}

|h2| Asymmetric Numeral Systems |h2e|

Asymmetric Numeral Systems (ANS) is a entropy encoding method used in data
compression developed by Jaroslaw Duda [2] in 2009.  It has a really simple
idea: take a message as a sequence of symbols and *encode it as a single
natural number* :math:`x`. 
If :math:`x` is small, it requires fewer bits to represent; if :math:`x` is
large, then it requires more bits to represent.  Or to think about it the other
way, if I can exploit the statistical properties of my message so that: (a) the
most likely messages get mapped to small natural numbers, and (b) the least likely
messages get mapped to larger natural numbers, then I will have achieved good
compression.  Let's explore this idea a bit more.

|h3| Encoding a Binary String to a Natural Number |h3e|

First off, let's discuss how we can even map a sequence of symbols to a natural
number.  To begin, we start with the simplest case: a sequence of binary symbols (0s and 1s).
We all know how to convert a binary string to a natural number, but let's break
it down into its fundamental parts.  We are particularly interested in how
to *incrementally* build up to the natural number by reading one bit at a time.

Suppose we have already converted some binary string :math:`b_1 b_2 b_3 \ldots b_i` 
(:math:`b_1` being the most significant digit) into a natural number
:math:`x_i` via the typical method of converting (unsigned) binary numbers to
natural numbers.  If we get another another binary digit :math:`b_{i+1}`, we
want to derive a coding function such that :math:`x_{i+1} = C(x_i, b_{i+1})`
generates natural number representation of :math:`b_1 b_2 b_3 \ldots b_{i+1}`.
If you remember from your discrete math courses, it should really just be
multiplying the original number by 2 (shifting up a digit in binary), and then
adding the new binary digit, which is just:

.. math::

    C(x_i, b_{i+1}) := 2x_i + b_{i+1} \tag{3}

If we start with :math:`x_0=0`, you can see that we'll be able to convert any
binary string iteratively (from MSB to LSB) to its natural number
representation.  Inversely, we can convert from any natural number to
iteratively recover the binary digit :math:`b_{i+1}` and the next resulting
natural number without that idgit, we can use the following decoding function:

.. math::

    (x_i, b_{i+1}) = D(x_{i+1}) := (\lfloor\frac{x_{i+1}}{2}\rfloor, x_{i+1} \bmod 2) \tag{4}

Nothing really new here but let's make a few observations:

* We shouldn't start with :math:`x_0=0`, because we won't be able to
  distinguish between "0", "00", "000" etc. because they all map to :math:`0`.
  Instead, let's start at :math:`x_0=1`, which effectively adds a leading "1"
  to each message we generate but now "0" and "00" can be distinguished as
  ("10" and "100").
* Let's look at how we're using :math:`b_{i+1}`.  In Equation 3, if
  :math:`b_{i+1}` is odd, then we add 1, else if even we add 0.  In Equation 4,
  we're doing the reverse, if the number :math:`x_{i+1}` is odd, we know we can
  recover a "1", else when even, we recover an "odd".  We'll use this idea in
  order to extend.
* Finally, the encoding using Equations 3 and 4 are optimal if we have a uniform
  distribution of "0"s and "1"s (i.e. :math:`p_0=p_1=\frac{1}{2}`).  Notice
  that the entropy :math:`H(x) = 2 \cdot \frac{1}{2}\log_2(\frac{1}{2}) = 1`,
  which results in 1 bit per binary digit -- exactly as we generate using these
  equations (if you exclude the fact that we start at 1).

The last two points are relevant because it gives us a hint as to how we might
extend this to non-uniform binary messages.  Our encoding is optimal because we
were able to spread the evens and odds (over any given range) in proportion to
their probability.  We'll explore this idea a bit more in the next section.

.. admonition:: Example 2: Encoding a Binary String to/from a Natural Number

    Using Equation 3 and 4, let's convert binary string 
    :math:`b_1 b_2 b_3 b_4 b_5 = 10011` to a natural number. 
    Starting with :math:`x_0=1`, we have:

    .. math::

        x_1 &= C(x_0, b_1) = 2x_0 + b_1 = 2(1) + 1 = 3 \\
        x_2 &= C(x_1, b_2) = 2x_1 + b_1 = 2(3) + 0 = 6 \\
        x_3 &= C(x_2, b_3) = 2x_2 + b_1 = 2(6) + 0 = 12 \\
        x_4 &= C(x_3, b_4) = 2x_3 + b_1 = 2(12) + 1 = 25 \\
        x_5 &= C(x_4, b_5) = 2x_4 + b_1 = 2(25) + 1 = 51 \\
        \tag{5}

    To recover our original messaage, we can use :math:`D(x_{i+1})`:

    .. math::

        (x_4, b_5) &= D(x_5) = (\lfloor\frac{x_{5}}{2}\rfloor, x_{5} \bmod 2) = 
            (\lfloor \frac{51}{2} \rfloor, 51 \bmod 2) = (25, 1) \\
        (x_3, b_4) &= D(x_4) = (\lfloor\frac{x_{4}}{2}\rfloor, x_{4} \bmod 2) = 
            (\lfloor \frac{25}{2} \rfloor, 25 \bmod 2) = (12, 1) \\
        (x_2, b_3) &= D(x_3) = (\lfloor\frac{x_{3}}{2}\rfloor, x_{3} \bmod 2) = 
            (\lfloor \frac{12}{2} \rfloor, 12 \bmod 2) = (6, 0) \\
        (x_1, b_2) &= D(x_2) = (\lfloor\frac{x_{2}}{2}\rfloor, x_{2} \bmod 2) = 
            (\lfloor \frac{6}{2} \rfloor, 6 \bmod 2) = (3, 0) \\
        (x_0, b_1) &= D(x_1) = (\lfloor\frac{x_{1}}{2}\rfloor, x_{1} \bmod 2) = 
            (\lfloor \frac{3}{2} \rfloor, 3 \bmod 2) = (1, 1) \\
        \tag{6}
    
    Notice that we recovered our original message in the reverse order.
    The number of bits needed to represent our natural number is :math:`\lceil
    \log_2(51) \rceil = 6` bits, which is just 1 bit above our ideal entropy of
    5 bits (assuming a uniform distribution).

|h3| Redefining the Odds (and Evens) |h3e|

Let's think about why the naive encoding in the previous section might result
in an optimal code for a uniform distribution.  For one, it spreads even and odd
numbers (binary strings ending in "0"'s and "1"'s respectively) uniformly across
any natural number range.  This kind of makes sense since they are uniformly
distributed.  What's the analogy for a non-uniform distribution?

If we were going to map a non-uniform distribution with 
:math:`p_1=p < 1-p = p_0`, then we would want the more frequent symbol (0 in
this case) to appear more often in any given mapped natural number range.  More
precisely, we would want even numbers to be mapped in a given range roughly 
:math:`\frac{1-p}{p}` more often than odd numbers.  Or stated another way,
in a given mapped natural number range from :math:`[1, x]` we would want to see
roughly :math:`x\cdot p` evens and :math:`x\cdot (1-p)` odds.
How can we achieve this?  Just define our coding function to do this! 

**SHOW DIAGRAM of even/odds asymmetrically distributed**

**Explain that if we didn't distribute it this way then **

For coding function :math:`C(x_i,b_{i+1})`, similarly defined as Equation 3,
we want: 

.. math::

    x_{i+1} = C(x_i,b_{i+1}) \approx \frac{x_i}{p_{b_{i+1}}} \tag{7}

**QUESTION: WHY is this equivalent to the density assumption?**

If we can guarantee this (with some reasonable approximation), then we can show that 
the encoding is roughly equal to the entropy (or the lower bound theoretical limit)
shown with the following reasoning:

.. admonition:: Entropy of Incrementally Adding a New Bit

    Assume natural number :math:`x_i` has :math:`\log_2(x_i)` bits of information
    that encodes some binary message.
    If we add a new bit :math:`b_{i+1}` to the message, then our encoding
    function would generate a new natural number :math:`x_{i+1} = C(x_i, b_{i+1})`.
    But the information contained in the new encoded message is:
    
    .. math::
        \log_2(x_{i+1}) = \log_2(C(x_i, b_{i+1}))  
                        \approx  \log_2(\frac{x_s}{p_{b_{i+1}}})
                        =\log_2(x) + \log_2(\frac{1}{p_{b_{i+1}}}) \tag{8}
    
    Hence, we added approximately :math:`\log_2(\frac{1}{p_{b_{i+1}}})` bits of
    information, which is precisely the entropy of the new bit.
  
As we can see, as long as we maintain the invariant of roughly :math:`x \cdot p` evens
(or equivalently :math:`x \cdot (1-p)` odds) as we go along, then we will have achieved
close to the theoretical compression limit.  In other words, if we "redefine" the frequency
of evens and odds, then we can achieve our goal.

In summary:

* A binary message encoded and decoded to a single natural number.
* Using this method, we can build an entropy encoder by defining a mapping of
  even and odd binary numbers (those ending in "1'/"0"s) in proportion to their
  probabilities (:math:`p, 1-p`) in a message.
* We can incrementally generate this number bit by bit by using a coding 
  function :math:`x_{i+1} = C(x_i, b_{i+1})` 
  (decoding function :math:`(x_i, b_{i+1}) = D(x_{i+1})`) that will
  iteratively generate a mapped natural number from (to) the previous mapped number
  and the next bit.

In the next sections, we'll discuss concrete implementations of these encoding
and decoding functions.

|h3| Uniform Binary Variant (uABS) |h3e|

Show :math:`\lceil (x+1)\cdot p \rceil - \lceil x\cdot p \rceil` is equivalent to what we want
of :math:`\lceil x\cdot p \rceil`

Show picture example of 2/5, 3/5 running from x=0,15, and color coating of even/odds and arrows to/from them

Describe mapping functions

Say it's equivalent when p=1/2

Add appendix of proof of mapping functions

|h3| Range variants (rANS) |h3e|

* rANS
* Renormalization

|h3| Other variants |h3e|

* tANS

|h2| Experiments |h2e|


|h3| Implementation |h3e|

* Need to be careful when implementing uABS -- floating point precision is not
  good enough b/c of rounding, also limited to N bits due to use of int64
* rANS, renormalization needs to pre-calculate lowerbound on compression

|h2| References |h2e|

* [1] "Lecture I: data compression ... data encoding", Jaroslaw Duda, Nokia Krakow, `<http://ww2.ii.uj.edu.pl/~smieja/teaching/ti/3a.pdf>`__ 
* [2] "Asymmetric numeral systems", Jarek Duda, `<https://arxiv.org/abs/0902.0271>`__
* [3] Wikipedia: `<https://en.wikipedia.org/wiki/Asymmetric_numeral_systems>`__

|h2| Appendix A: Proof of Floor/Ceil |h2e|


