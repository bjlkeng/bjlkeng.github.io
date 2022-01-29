.. title: Lossless Compression with Asymmetric Numeral Systems
.. slug: lossless-compression-with-asymmetric-numeral-systems
.. date: 2020-09-26 10:37:43 UTC-04:00
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

During my undergraduate days, one of the most interesting courses I took was on
coding and compression.  Here was a course that combined algorithms,
probability and secret messages, what's not to like? [1]_ I ended up not going
down that career path, at least partially because communications systems had
its heyday around the 2000s with companies like Nortel and Blackberry and its
predecessors (some like to joke that all the major theoretical breakthroughs
were done by Shannon and his discovery of information theory around 1950).  Fortunately, I
eventually wound up studying industrial applications of classical AI techniques
and then machine learning, which has really grown like crazy in the last 10
years or so.  Which is exactly why I was so surprised that a *new* and *better*
method of lossless compression was developed in 2009 *after* I finished my
undergraduate degree when I was well into my PhD.  It's a bit mind boggling that
something as well-studied as entropy-based lossless compression still had
(have?) totally new methods to discover, but I digress.

In this post, I'm going to write about a relatively new entropy based encoding
method called Asymmetrical Numeral Systems (ANS) developed by Jaroslaw (Jarek)
Duda [2].  If you've ever heard of Arithmetic Coding (probably best known for
its use in JPEG compression), ANS runs in a very similar vein.  It can
generate codes that are close to the theoretical compression limit
(similar to Arithmetic coding) but is *much* more efficient.  It's been used in 
modern compression algorithms since 2014 including compressors developed
by Facebook, Apple and Google [3].  As usual, I'm going to go over some
background, some math, some examples to help with intuition, and finally some
experiments with a toy ANS implementation I wrote.  I hope you're as
excited as I am, let's begin!

.. TEASER_END

|h2| Background: Data Coding and Compression |h2e|

Most modern digital communication is done via transmitting **bits** 
(i.e. 0/1 values) with the goals of both reliability and efficiency.
A **symbol** is unit of information that is transmitted, which can be the bits itself
or a higher level concept represented by a sequence of bits (e.g. "a", "b", "c", "d" etc.).
An **alphabet** is the set of all symbols that you can transmit.
A **message** is a sequence of symbols that you want to transmit.

Often you will want to **code** a message (such as a file) for storage or
transmission on a communication channel by specifying some rules on how to
transform it into another form.  The usual reasons why you want to code a message is
compression (fewer bits), redundancy (error correction/detection), or
encryption (confidentiality).  For compression, we generally have two main
approaches: lossy and lossless.  In lossy schemes, you drop some non-essential
details in the message (think image compression) to trade-off a greater
reduction in size.  This trick is used extensively in things like image 
(e.g. JPEG) or audio (e.g. MP3) compression.  Lossless schemes on the other
hand aim to retain the exact message reducing the file size by exploiting some
statistical redundancies in the data.  We're mainly talking about lossless
compression schemes today.

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
sequence of a character by the character and how many times it repeats.  In both
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
    theorem) is:

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
number.  We can start with the simplest case: a sequence of binary symbols (0s and 1s).
We all know how to convert a binary string to a natural number, but let's break
it down into its fundamental parts.  We are particularly interested in how
to *incrementally* build up to the natural number by reading one bit at a time.

Suppose we have already converted some binary string :math:`b_1 b_2 b_3 \ldots b_i` 
(:math:`b_1` being the most significant digit) into a natural number
:math:`x_i` via the typical method of converting (unsigned) binary numbers to
natural numbers.  If we get another another binary digit :math:`b_{i+1}`, we
want to derive a coding function such that :math:`x_{i+1} = C(x_i, b_{i+1})`
generates the natural number representation of :math:`b_1 b_2 b_3 \ldots b_{i+1}`.
If you remember your discrete math courses, it should really just be
multiplying the original number by 2 (shifting up a digit in binary), and then
adding the new binary digit, which is just:

.. math::

    C(x_i, b_{i+1}) := 2x_i + b_{i+1} \tag{3}

If we start with :math:`x_0=0`, you can see that we'll be able to convert any
binary string iteratively (from MSB to LSB) to its natural number
representation.  Inversely, we can convert from any natural number to
iteratively recover both the binary digit :math:`b_{i+1}` and the next
resulting natural number without that digit using the following decoding
function:

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
  recover a "1", else when even, we recover an "0".  We'll use this idea in
  order to extend to more complicated cases.
* Finally, the encoding using Equations 3 and 4 are optimal if we have a uniform
  distribution of "0"s and "1"s (i.e. :math:`p_0=p_1=\frac{1}{2}`).  Notice
  that the entropy :math:`H(x) = -2 \cdot \frac{1}{2}\log_2(\frac{1}{2}) = 1`,
  which results in 1 bit per binary digit, which is exactly what these
  equations generate  (if you exclude the fact that we start at 1).

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
        x_2 &= C(x_1, b_2) = 2x_1 + b_2 = 2(3) + 0 = 6 \\
        x_3 &= C(x_2, b_3) = 2x_2 + b_3 = 2(6) + 0 = 12 \\
        x_4 &= C(x_3, b_4) = 2x_3 + b_4 = 2(12) + 1 = 25 \\
        x_5 &= C(x_4, b_5) = 2x_4 + b_5 = 2(25) + 1 = 51 \\
        \tag{5}

    To recover our original message, we can use :math:`D(x_{i+1})`:

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
:math:`p_1=p < 1-p = p_0`, then we would want the more frequent symbol 
(0 in this case) to appear more often in any given mapped natural number range.
More precisely, we would want even numbers to be mapped in a given range
roughly :math:`\frac{1-p}{p}` more often than odd numbers.  Or stated another
way, in a given mapped natural number range from :math:`[1, N]` we would want
to see roughly :math:`N\cdot p` evens and :math:`N\cdot (1-p)` odds.
This is the right intuition but doesn't really show how it might generate an
optimal code.  Let's work backwards from an optimal compression scheme and
figure out what we would need.

We are trying to define the encoding function :math:`x_{i+1} = C(x_i, b_{i+1})`
(similarly to Equation 3) such that each incremental bit generates the minimal 
amount of entropy.  Assuming that :math:`x_i` has :math:`\log_2 (x_i)` bits of
information, and we want to encode :math:`b_{i+1}` optimally with
:math:`-\log_2(p_{b_{i+1}})` bits, we have (with a bit of abuse of entropy notation):

.. math::
    H(x_{i+1}) &= H(C_{\text{opt}}(x_i, b_{i+1})) \\
               &= H(x_i) + H(b_{i+1})\\
               &= \log_2(x_i) - \log_2(p_{b_{i+1}})\\
               &= \log_2(\frac{x_i}{p_{b_{i+1}}}) \\
    &\implies C_{\text{opt}}(x_i, b_{i+1}) \approx \frac{x_i}{p_{b_{i+1}}} 
        \tag{8}

Therefore, if we can define :math:`C(x_i, b_{i+1}) \approx \frac{x_i}{p_{b_{i+1}}}`
then we will have achieved an optimal code!  Let's try to understand what this
mapping means.

From Equation 8, if we are starting at some :math:`x_i` and get a new bit
:math:`b_{i+1}=1` (an odd number), then :math:`x_{i+1}\approx\frac{x_i}{p}`.
But we know :math:`x_i` can be any natural number, so this implies that 
odd numbers will be placed at (roughly), :math:`\frac{1}{p}, \frac{2}{p},
\frac{3}{p}, \ldots` intervals for any given natural number.
This also means, we'll see an odd number
(roughly) every :math:`\frac{1}{p}` natural numbers.  
But if we take a closer look, this is precisely the condition of having roughly
:math:`N\cdot p` for the first :math:`N` natural numbers (:math:`\text{# of
Odds} = N / \frac{1}{p} = N\cdot p`).  Similarly, we'll see even numbers
(roughly) every :math:`\frac{1}{1-p}`, which also means we'll see (roughly)
:math:`N \cdot (1-p)` in the first :math:`N` natural numbers.  So our intuition
does lead us towards the solution of an optimal code after all!

.. figure:: /images/ans_even_odd.png
  :height: 200px
  :alt: Distribution of Evens and Odds for Various :math:`p`
  :align: center

  **Figure 1: Distribution of Evens and Odds for Various :math:`p`**


Thinking about this code a bit differently, we are essentially redefining the
frequency of evens and odds with this new mapping.  We can see this more
clearly in Figure 1.  For different values of :math:`p`, we can see a repeating
pattern of where the evens and odds fall.  When :math:`p=1/2`, we see an
alternating pattern (never mind that :math:`2` is mapped to an odd, this is an
unimportant quirk of the implementation) as we usually expect.  However,
when we go to non-uniform distributions, we can see repeating but
non-alternating patterns.  One thing you may notice is that the above equations
are in :math:`\mathbb{R}` but we need them to mapped to natural numbers!
Figure 1 implicitly does some of the required rounding and we'll see more of
that in the implementations below.

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
* If we can guarantee our coding function :math:`C(x_i, b_{i+1}) \approx \frac{x_i}{p_{b_{i+1}}}`
  then we will have achieved an optimal code.

|h3| Uniform Binary Variant (uABS) |h3e|

Without loss of generality, let's use a binary alphabet with odds ending in "1" and evens
ending in "0", and :math:`p_1 = p < 1-p = p_0` (odds are always less frequent than evens).
We know we want approximately :math:`N\cdot p` odd numbers mapped in the first
:math:`N` mapped natural numbers.  Since we have to have a non-fractional
number of odds, let's pick :math:`\lceil N \cdot p \rceil` odds in the first
:math:`N` mapped natural numbers.  From this, we get this relationship for
any given :math:`N` and :math:`N+1` (try to validate it against Figure 1):

.. math::

    \lceil (N+1)\cdot p \rceil - \lceil N\cdot p \rceil 
    = \left\{
        \begin{array}{ll}
            1 && \text{ if } N \text{ has an odd mapped} \\
            0 && \text{otherwise} \\
        \end{array}
    \right. \tag{9}

Another way to think about it is: if we're at :math:`N` and we've filled our :math:`\lceil N \cdot p\rceil`
odd number "quota" then we don't need to see another odd at :math:`N+1` (the :math:`0` case).
Conversely, if going to :math:`N+1` makes it so we're behind our odd number
"quota" then we should make sure that we map an odd at :math:`N` (the :math:`1` case).

Now here's the tricky part: what coding function :math:`x_{i+1} = C(x_i, b_{i+1})` 
satisfies Equation 9 (where :math:`x_i` is our mapped natural number)?
It turns out this one does:

.. math::
    
    C(x_i, b_{i+1})
    = \left\{
        \begin{array}{ll}
            \lceil \frac{x_i+1}{1-p} \rceil - 1 && \text{if } b_{i+1} = 0 \\
            \lfloor \frac{x_i}{p} \rfloor && \text{otherwise} \\
        \end{array}
    \right. \tag{10}

I couldn't quite figure out a sensible derivation of why this particular
function works but it's probably non-trivial.  The main problem is
that we're working with natural numbers, so dealing with floor and ceil
operators is tricky.  Additionally, Equation 9 kind of looks like a 
some kind of `difference equation <https://en.wikipedia.org/wiki/Linear_difference_equation>`__,
which are generally very difficult to solve.  However, I did manage to
prove that Equation 10 is consistent with Equation 9.  See Appendix A for the
proof.

Using Equation 10, we can now code any binary message using the same method we used
in the previous section with Equation 3: iteratively applying Equation 10 one
bit at a time.  The matching decoding function is essentially the reverse calculation:

.. math::
    
    (x_i, b_{i+1}) &= D(x_{i+1}) \\
    b_{i+1} &= \lceil (x_{i+1}+1)\cdot p \rceil - \lceil x_{i+1}\cdot p \rceil  \\
    x_i &= \left\{
        \begin{array}{ll}
            x_{i+1} - \lceil x_{i+1} \cdot p \rceil && \text{if } b_{i+1} = 0 \\
            \lceil x_{i+1} \cdot p \rceil && \text{otherwise} \\
        \end{array}
    \right. \tag{11}

The decoding of a bit is calculated exactly as we have designed it in Equation 9,
and depending on which bit was decoded, we perform the reverse calculation of
Equation 10.  For the :math:`b_{i+1} = 0` case, it may not look like the reverse
calculation but the math should work out (haven't proven it, but my
implementation works).  In the end, the equations to encode/decode are straight
forward but the logic of arriving at them is far from it.

.. admonition:: Example 3: Encoding a Binary String to/from a Natural Number using uABS

    Using the same binary string as Example 2, 
    :math:`b_1 b_2 b_3 b_4 b_5 = 10011`, let's encode it using uABS but
    with :math:`p=\frac{7}{10}` (recall we assume that :math:`p=p_1 < p_0=1-p`).
    Using Equation 10 and starting with :math:`x_0=1`, we get:

    .. math::

        x_1 &= C(x_0, b_1) = \lfloor \frac{x_0}{p} \rfloor = \lfloor 1\cdot \frac{10}{3} \rfloor = 3 \\
        x_2 &= C(x_1, b_2) = \lceil \frac{x_i+1}{1-p} \rceil - 1 = \lceil (3+1)\frac{10}{7} \rceil - 1 = 5 \\
        x_3 &= C(x_2, b_3) = \lceil \frac{x_i+1}{1-p} \rceil - 1 = \lceil (5+1)\frac{10}{7} \rceil - 1 = 8 \\
        x_4 &= C(x_3, b_4) = \lfloor \frac{x_0}{p} \rfloor = \lfloor 8\cdot \frac{10}{3} \rfloor = 26 \\
        x_5 &= C(x_4, b_5) = \lfloor \frac{x_0}{p} \rfloor = \lfloor 26\cdot \frac{10}{3} \rfloor = 86 \\
        \tag{12}

    Decoding can be applied in a similar way with Equation 11, which recovers
    our original message of "10011" (but in reverse order):

    .. math::

        b_5 &= \lceil (x_5+1)\cdot p \rceil - \lceil x_5\cdot p \rceil 
             = \lceil (86+1)\cdot \frac{3}{10} \rceil - \lceil 86\cdot \frac{3}{10} \rceil 
             = 1 \\
        x_4 &= \lceil x_5 \cdot p \rceil 
             = \lceil 86\cdot \frac{3}{10} \rceil 
             = 26 \\
        b_4 &= \lceil (x_4+1)\cdot p \rceil - \lceil x_4\cdot p \rceil 
             = \lceil (26+1)\cdot \frac{3}{10} \rceil - \lceil 26\cdot \frac{3}{10} \rceil 
             = 1 \\
        x_3 &= \lceil x_4 \cdot p \rceil 
             = \lceil 26\cdot \frac{3}{10} \rceil 
             = 8 \\
        b_3 &= \lceil (x_3+1)\cdot p \rceil - \lceil x_3\cdot p \rceil 
             = \lceil (8+1)\cdot \frac{3}{10} \rceil - \lceil 8\cdot \frac{3}{10} \rceil 
             = 0 \\
        x_2 &= x_3 - \lceil x_3 \cdot p \rceil 
             = 8 - \lceil 8\cdot \frac{3}{10} \rceil 
             = 5 \\
        b_2 &= \lceil (x_2+1)\cdot p \rceil - \lceil x_2\cdot p \rceil 
             = \lceil (5+1)\cdot \frac{3}{10} \rceil - \lceil 5\cdot \frac{3}{10} \rceil 
             = 0 \\
        x_1 &= x_2 - \lceil x_2 \cdot p \rceil 
             = 5 - \lceil 5\cdot \frac{3}{10} \rceil 
             = 3 \\
        b_1 &= \lceil (x_1+1)\cdot p \rceil - \lceil x_1\cdot p \rceil 
             = \lceil (3+1)\cdot \frac{3}{10} \rceil - \lceil 3\cdot \frac{3}{10} \rceil 
             = 1 \\
        x_0 &= \lceil x_1 \cdot p \rceil 
             = \lceil 3\cdot \frac{3}{10} \rceil 
             = 1 \\
        \tag{13}

    Another popular way to visualize this is using a tabular method in Figure 2.
    In the top row, we have the same visualization of evens/odds as Figure 1 for :math:`p=\frac{3}{10}`,
    which is essentially :math:`C(x_i, b_{i+1})`.
    In the second and third row, it shows which numbers are mapped to
    evens/odds and counts the number of "slots" of evens/odds we have see up to
    that point.
    So for :math:`C(x_i, b_{i+1})=3`, it's mapped to the first odd "slot", and
    for :math:`C(x_i, b_{i+1})=26`, it's mapped to the eighth odd "slot".  The
    same thing happens on the even side.  

    .. figure:: /images/ans_ex3.png
      :height: 150px
      :alt: Tabular Visualization of uABS Encoding
      :align: center
    
      **Figure 2: Tabular Visualization of uABS Encoding**

    This turns out to be precisely what Equation 10 is doing: for any given
    :math:`x_i` it's trying to find the next even/odd "slot" to put
    :math:`x_{i+1}` in.
    The yellow lines trace out what an encoding for "10011" would look like.
    Our current number :math:`x_i` along with the incoming bit :math:`b_{i+1}`
    defines which "slot" we should go in (the diagonal arrows), and Equation 10
    calculates the next natural number associated with it (the "up" arrows).
    Decoding would follow a similar process but in reverse.

|h3| Range Variant (rANS) |h3e|

We saw that uABS works on a binary alphabet, but we can also apply the same concept
to an alphabet of any size (with some modifications). The first thing to notice
is that that the argument from Equation 8 works (more or less) with *any*
alphabet, not just binary ones (just replace the bit :math:`b_{i+1}` with symbol :math:`s_{i+1}`).
That is, adding an incremental symbol (instead of a bit) should only increase
the total entropy of the message by the entropy of that symbol.  Equation 8
would only need to reference symbol and the same logic would work.  

Another problem are those pesky real numbers.  Theoretically, we can have
arbitrary real numbers for the probability distribution of our alphabet.  We
"magically" found a nice formula in Equation 10/11 that encodes/decodes any
arbitrary :math:`p`, but in the case of a larger alphabet, it's a bit tougher.
Instead, a restriction that we'll place is that we'll quantize the probability
distribution in :math:`2^n` chunks.  So :math:`p_s\approx \frac{f_s}{2^n}`,
where :math:`f_s` is a natural number.
This quantization of the probability distribution, simplifies things for us by
allowing us to have a simpler and more efficient coding/decoding function
(although it's not clear to me if it's possible to do it without quantization).

Instead of our previous idea of evens and odds, what we'll be doing is extending this idea 
and "coloring" each number.  So for an alphabet of size 3, we might color
things red, green and blue.  Figure 3 shows a few examples with this alphabet
with :math:`n=3` quantization for a few different distributions (this is analogous
to Figure 1).

.. figure:: /images/ans_rans.png
  :height: 200px
  :alt: Distribution of "blue", "green" and "red" symbols
  :align: center

  **Figure 3: Distribution of "blue", "green" and "red" symbols**

So how does it work?  It's not too far off from uABS, we use the following equations to encode/decode:

.. math::

    C(x_i, s_{i+1}) &= \lfloor \frac{x_i}{f_s} \rfloor \cdot 2^n + CDF[s]  \tag{14} + (x_i \bmod f_s)  \\
    s_{i+1} &= \text{symbol}(x_{i+1} \bmod 2^n) \text{ such that } CDF[s] \leq x_{i+1} \bmod 2^n < CDF[s+1] \tag{15} \\
    x_i = D(x_{i+1}) &= f_s \cdot \lfloor x_{i+1} / 2^n \rfloor - CDF[s] + (x_{i+1} \bmod 2^n) \tag{16}

Where :math:`CDF[s] := f_0 + f_1 + \ldots + f_{s-1}`, essentially the
cumulative distribution function for a given ordering of the symbols.  You'll notice
that since we've quantized the distribution in terms of powers of 2, we can replace
the multiplications, divisions and modulo with left shifting, right shifting
and logical masking, respectively, which makes this much more efficient computationally.

The intuition for Equation 14-16 isn't too far from from uABS: for a given
:math:`N`, we want to maintain the property that we roughly see 
:math:`N\cdot  p_s = N \cdot \frac{f_s}{2^n}` of symbols :math:`s`. Looking
at Equation 14, we can see how it accomplishes this:

* :math:`\lfloor \frac{x_i}{f_s} \rfloor \cdot 2^n`: finds the right :math:`2^n` range
  (recall that we have a repeating pattern every :math:`2^n` natural numbers).
  If :math:`f_s` is small, say :math:`f_s=1`, then it only appears once every
  :math:`2^n` range.  If :math:`f_s` is large, then we would expect to see
  :math:`f_s` numbers mapped to every :math:`2^n` range.
* :math:`CDF[s]` finds the offset within the :math:`2^n` range for the current
  symbol :math:`s` -- all :math:`s` symbols will be grouped together within
  this range starting here.
* :math:`(x_i \bmod f_s)` finds the precise location within this sub-range
  (which has precisely :math:`f_s` spaces allocated for it).

The decoding is basically just the reverse operation of the encoding.

Since we maintain this repeating pattern, we implicitly are maintaining the
property that we'll see :math:`x_i \cdot p_s` ":math:`s`" symbols within the
first :math:`x_i` natural numbers.  

.. admonition:: Example 4: Encoding a Ternary String to/from a Natural Number using rANS

    Using the alphabet ['a', 'b', 'c'] with quantization :math:`n=3` and distribution
    :math:`[f_a, f_b, f_c]=[5, 2, 1]` (:math:`CDF[s] = [0, 5, 7, 8]`), let's encode the string "abc".
    We need to start with :math:`x_0=8` or we won't be able to encode repeated
    values of 'a' (similar to how we start uABS at 1). In fact, we just need to
    start with the :math:`\max f_s` but to be safe, we'll use :math:`2^n`.
    Using Equation 14:

    .. math::

       x_1 &= C(x_0, a) 
            = \lfloor \frac{x_0}{f_a} \rfloor \cdot 2^3 + CDF[a] + (x_0 \bmod f_a) 
            = \lfloor \frac{8}{5} \rfloor \cdot 8 + 0 + (8 \bmod 5) 
            = 11 \\
       x_2 &= C(x_0, b)
            = \lfloor \frac{x_1}{f_b} \rfloor \cdot 2^3 + CDF[b] + (x_1 \bmod f_b) 
            = \lfloor \frac{11}{2} \rfloor \cdot 8 + 5 + (11 \bmod 2)
            = 46  \\
       x_3 &= C(x_0, c)
            = \lfloor \frac{x_2}{f_c} \rfloor \cdot 2^3 + CDF[c] + (x_2 \bmod f_c)
            = \lfloor \frac{46}{1} \rfloor \cdot 8 + 7 + (46 \bmod 1)
            = 375 \\
        \tag{17}

    Decoding, works similarly using Equation 15-16:

    .. math::

        s_2 &= \text{symbol}(x_3 \bmod 8) = \text{symbol}(375 \bmod 8) = c \\
        x_2 &= D(x_3) 
             = f_c \cdot \lfloor x_3 / 8 \rfloor - CDF[c] + (x_3 \bmod 8)
             = 1 \cdot \lfloor 375 / 8 \rfloor - 7  + (375 \bmod 8)
             = 46 \\
        s_1 &= \text{symbol}(x_2 \bmod 8) = \text{symbol}(46 \bmod 8) = b \\
        x_1 &= D(x_2) 
             = f_b \cdot \lfloor x_2 / 8 \rfloor - CDF[b] + (x_2 \bmod 8)
             = 2 \cdot \lfloor 46 / 8 \rfloor - 5  + (46 \bmod 8)
             = 11 \\
        s_0 &= \text{symbol}(x_1 \bmod 8) = \text{symbol}(11 \bmod 8) = a \\
        x_0 &= D(x_1) 
             = f_a \cdot \lfloor x_1 / 8 \rfloor - CDF[a] + (x_1 \bmod 8)
             = 5 \cdot \lfloor 11 / 8 \rfloor - 0 + (11 \bmod 8)
             = 8 \\
        \tag{18}

    We can build the same table as Figure 2 except we'll have four rows:
    for :math:`C(x_i, s_{i+1}), a, b, c`.  Building the table is left as
    an exercise for the reader :)

.. admonition:: Note about the starting value of :math:`x_0`

    In Example 4, we started on :math:`x_0=2^n`.  This is because if we didn't,
    we could get into the situation where we couldn't distinguish certain
    repetitions of strings such as: [a, aa, aaa], for example.  Using Example 4, 
    let's see what we'd get starting with :math:`x_0=1`:

    .. math::

       x_1 &= C(x_0, a) 
            = \lfloor \frac{x_0}{f_a} \rfloor \cdot 2^3 + CDF[a] + (x_0 \bmod f_a)
            = \lfloor \frac{1}{5} \rfloor \cdot 8 + 0 + (1 \bmod 5)
            = 1 \\ 
       x_2 &= C(x_1, a) 
            = \lfloor \frac{x_0}{f_a} \rfloor \cdot 2^3 + CDF[a] + (x_0 \bmod f_a)
            = \lfloor \frac{1}{5} \rfloor \cdot 8 + 0 + (1 \bmod 5)
            = 1 \\     
       x_3 &= C(x_2, a) 
            = \lfloor \frac{x_0}{f_a} \rfloor \cdot 2^3 + CDF[a] + (x_0 \bmod f_a)
            = \lfloor \frac{1}{5} \rfloor \cdot 8 + 0 + (1 \bmod 5)
            = 1 \\
        \tag{19}

    As you can see we get nowhere fast.  The reason is that the first term
    always rounds down, resulting in the exact same value.  Similarly, the
    second term always resolves the same thing (since 'a' is the first symbol
    in our ordering), and the third term as well.
    
    I think (haven't really proven it) that the safest option is to have
    :math:`\max f_s` as your starting value.  This will ensure that the first
    term will always be >= 0, resulting in a different number than you started
    with.  To be safe, :math:`2^n > \max f_s`, which is just a bit nicer.
    In some sense, we're "wasting" the initial numbers here starting :math:`x_0`
    larger but it's necessary in order to encode repeated strings and handle
    these corner cases.  
    
    Another way you could go about it, is that do a fixed mapping for the first
    :math:`2^n` numbers (a base case of sorts), and then from there you can
    apply the formula.  I didn't try this but I think that this is also
    possible.

|h3| Renormalization |h3e|

The astute reader may have already been wondering how this can work in practice.
It works great when you only have a message of length five or so, but what about a
1 MB file?  If we use a 1-byte=256-length alphabet, we could potentially be
getting a number on the order of :math:`2^{1000000n}` over this 1M-length string.
Surely no integer type will be able to efficiently handle that!
It turns out there is a simple trick to ensure that :math:`x^i \in [2^M, 2^{2M} - 1]`.

The idea is that during encoding once :math:`x_i` gets too big, we simply write
out the lower :math:`M` bits to ensure it stays between :math:`[2^M, 2^{2M} - 1]`
(e.g. :math:`M=16` bits).
Similarly, during decoding, if :math:`x_i` is too small, shift the current
number up and read in :math:`M` bits into the lower bits.  As long as you take
care to make sure each operation is symmetric, it should allow you to always
play with a number that fits within an integer type.
    
**Listing 1: Encoding and Decoding rANS Python Pseudocode with Renormalization**

.. code:: python 
    :linenos:

    MASK = 2**M - 1 
    BOUND = 2**(2*M) - 1

    # Encoding
    s = readSymbol()
    x_test = (x / f[s]) << n + (x % f[s]) + c[s]
    if (x_test > BOUND):
        write16bits(x & MASK)
        x = x >> M
    x = (x / f[s]) << n + (x % f[s]) + c[s]

    # Decoding
    s = symbol[x & MASK]
    writeSymbol(s)
    x = f[s] (x >> n) + (x & MASK) - c[s]
    if (x < 2**M):
        x = x << M + read16bits()

Listing 1 shows the Python pseudo code for rANS encoding and decoding with
renormalization.  Notice that we use Equation 14-16 but with more efficient
bit-wise operations.

|h3| Other variants |h3e|

As you can imagine, there are numerous variants of the above
algorithms/concepts, especially as it relates to efficient implementations.
One of the most practical is one called 
`tANS <https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Tabled_variant_(tANS)>`__ 
or the tabled variant.  In this variation, we build a finite state machine
(i.e. table) to pre-compute all the calculations we would have done in rANS.
This has a bit more upfront cost but will make the encoding/decoding much faster
without the need for multiplications.

Another extension of tANS is the ability to encrypt the message directly in the
tANS algorithm.  Since we're building a table, we don't really need to maintain
Equation 14-16 but rather can pick any repeating pattern.  So instead of the
typical rANS repeating pattern, we can scramble it based on some random number.
See [1] for more details.

|h2| Implementation Details |h2e|

I implemented some toy versions of uABS and rANS in Python which you can find on my 
`Github <https://github.com/bjlkeng/sandbox/tree/master/ans>`__.
Surprisingly, it was a bit trickier than I thought due to a few gotchas.
Here are some notes for implementing uABS:

* Python's integer type is theoretically unlimited but I used some `numpy`
  functions, which *do* have a limited range (64-bit integer).  You can see
  this when using uABS with large binary strings, particularly with close to
  uniform distributions, where the code encodes/decodes string incorrectly.
* The other "gotcha" is that with uABS, we are actually (sort of) dealing with
  real numbers, which is a poor match for floating point data types.  Python's
  floating point type definitely *has* limited precision, so the usual problems
  of being not represent real numbers exactly become a problem.  Especially
  when we need to apply ceil/floor where a `0.000001` difference is meaningful.
  To hack around this, I simply just wrapped everything in Python's `Decimal`
  type.
* Finally, to ensure that the smaller :math:`p` was always mapped to the "1",
  I had do some swapping of the characters and probabilities.

For rANS, it was a bit easier, except for the renormalization, where I had to
play around a bit:

* I decided to simplify my life, I would just directly take the frequencies
  (:math:`f_s`) along with the quantization bits (:math:`n`) instead of 
  introducing errors quantizing things myself.
* As mentioned above, I kept having errors until I figured out that :math:`x_0`
  needed to start at a large enough value.  With renormalization, I start
  it at :math:`x_0=2^M-1` since we want :math:`x_i \in [2^M, 2^{2M}-1]`.
  Turns out you need to have minus one there or else you get into a corner case
  where the decoding logic stops decoding early (or at least my implementation did).
* The other thing I had to "figure out" was the `BOUND` in Listing 1.  I initially
  thought it was simply just :math:`2^{2M}-1` but I was wrong.  In [1], they 
  reference a `bound[s]` variable that is never defined, so I had to reverse
  engineer it.  I'm almost positive there is a better way to do it than what
  I have in Listing 1, but I think my way is the most straight forward.
* In the decoding, there is a step where you have to lookup which symbol was
  decoded.  I simply used `numpy.argmax`, which I presume does a linear search.
  Apparently, this is one place where you can do something smarter but I wasn't
  too interested in this part.
* I didn't have to do any of the funny wrapping using `Decimal` that I did with
  uABS because of the quantization to :math:`n` bits.  There is still a
  division and call to `floor()` but since we're dealing with integers in the
  division, the chances of causing issues is pretty small I think (at least I
  haven't seen it yet).

Finally, none of my toy implementation, nor the compression values are quite
realistic because you also need to include the encoding of the probability
distribution itself!  Something that you would surely include as metadata in a
file.  However, if we're compression a file that's relatively big, this constant
amount of data *should* be negligible.

|h2| Experiments |h2e|

The setup for uABS and rANS experiments were roughly the same.  First, a
random strings of varying length was generated based on the alphabet,
distribution and quantization bits (for rANS).  Next, the compression algorithm
is run against the string and the original message size, ideal (Shannon limit) size,
and actual size were measured or calculated.  For each uABS experiment, each
setting was run with 100 different strings and averaged, while for rANS it was
run 50 times and averaged.

Figure 4 shows the results for uABS where "actual_ratio" stands for compression
ratio.  First off, more skewed distributions (lower :math:`p`) result in higher
compression.  This is sensible because we can exploit the fact that odd numbers
appear much more often.  Next, it's clear that as the message length increase,
we get a better compression ratio (closer to ideal).  This expected as the
asymptotic behavior of the code starts paying off.  Interesting, for more
skewed distributions (:math:`p=0.01`), it takes much longer message lengths for
us to get close to the theoretical limit.  We would probably need a message
length of :math:`N * 1 / p` to start approaching that limit.  Unfortunately, since
I didn't implement renormalization, I couldn't push the message length too much
further since the numbers got too big.

.. figure:: /images/ans_uabs.png
  :alt: Experimental Results
  :align: center

  **Figure 4: Experimental Results for uABS (dashed lines are the ideal compression ratio)**

Figure 5 show the first set of results for rANS.  Here we used an 256 character
alphabet (8-bits = 1byte) with 15 quantization bits and 24 renormalization
bits.   Figure 5 shows various distributions for varying message lengths.
Uniform is self explanatory, `power_X` are normalized power distributions with
exponent :math:`X`.  We see the same pattern of more skewed distributions
having higher compression and reaching close to theoretical limit with longer
message sizes.

.. figure:: /images/rans_msg_len.png
  :alt: Experimental Results
  :align: center

  **Figure 5: Experimental Results for rANS varying message length and distribution (dashed lines are the ideal compression ratio)**

Figure 6 shows an ablation study for rANS on quantization bits.  I held constant
`power_50` distribution with message length 1000 and varied quantization bits
and renormalization bits.  `renormalization_bits = add_renorm_bits + quantization_bits`.
We can see that more precise quantization yields better compression, as
expected.  It can get closer to the actual `power_50` distribution instead of
being a coarse approximation. Varying the renormalization bits relative to quantization
doesn't seem to have much effect in terms of compression ratio (I suspect there's more to 
it here but I didn't want to spend too much time investigating it).

.. figure:: /images/rans_quantization.png
  :alt: Experimental Results
  :align: center

  **Figure 6: Experimental Results for rANS varying quantization bits and renormalization bits**

|h2| Conclusion |h2e|

Well this post was definitely another tangent that I went off on.  In fact, the
post I actually wanted to write was ML related but I got side tracked trying to
understand ANS.  It just was so interesting that I thought I should learn it
more in depth and write a post on it.  I keep trying to make more time for
writing on this blog but I always seem to have more and more things keeping me
busy professionally and personally (which is a good thing!).  Anyways, look out
for a future post where I will make reference to ANS.  Thanks for reading!

|h2| References |h2e|

* [1] "Lecture I: data compression ... data encoding", Jaroslaw Duda, Nokia Krakow, `<http://ww2.ii.uj.edu.pl/~smieja/teaching/ti/3a.pdf>`__ 
* [2] "Asymmetric numeral systems", Jarek Duda, `<https://arxiv.org/abs/0902.0271>`__
* [3] Wikipedia: `<https://en.wikipedia.org/wiki/Asymmetric_numeral_systems>`__

|h2| Appendix A: Proof of uABS Coding Function |h2e|

(*As an aside: I spent longer than I'd like to admit trying to figure out this proof.
It turns out that trying to prove things involving floor and ceil functions wasn't
so obvious for a computer engineer by training.
I tried looking up a bunch of identities and going in circles using modulo 
notation without much success.  It was only after going back to the definition
of the floor/ceil operators, did I figure out the proof below.  There's
probably some lesson here about first principles but I'll let you take what you 
want from this story.*)

Let's start out by assuming that :math:`p = \frac{a}{b}` can be represented as
a rational number for some relatively prime :math:`a, b \in \mathbb{Z}^{+}`.
Practically, we're working with non-infinite precision, so it's not too big of
a stretch.  To verify that Equation 10 is consistent with Equation 9, we'll use
substitution and show that the two equations are consistent.

**Case 1: N is odd**

Re-write Equation 9 odd case:

.. math::

    x_{i+1} = \lfloor \frac{x_i}{p} \rfloor = \lfloor \frac{bx_i}{a} \rfloor = \frac{bx_i}{a} - \frac{m}{a} && \text{for some } 0 \leq m < a, m \in \mathbb{Z} \\
    \tag{A.1}

Substitute Equation A.1 into Equation 9 (where we're taking :math:`N=x_{i+1}`):

.. math::

    \lceil (N+1)\cdot p \rceil - \lceil N\cdot p \rceil 
    &= \lceil (\frac{bx}{a} - \frac{m}{a} + 1)\frac{a}{b} \rceil - \lceil (\frac{bx}{a} - \frac{m}{a})\frac{a}{b}  \rceil \\
    &= \lceil x - \frac{m}{b} + \frac{a}{b} \rceil -  \lceil x - \frac{m}{b} \rceil \\
    &= x + \lceil - \frac{m}{b} + \frac{a}{b} \rceil -  x - \lceil - \frac{m}{b} \rceil && \text{ since } x \in \mathbb{Z} \\
    &= \lceil \frac{a-m}{b} \rceil -  \lceil - \frac{m}{b} \rceil  && \text{ since } 0 \leq m < a < b \\
    &= 1 - 0 = 1
    \tag{A.2}

**Case 2: N is even**

Substitute Equation 9 and A.3 into Equation 9:

.. math::

    \lceil (N+1)\cdot p \rceil - \lceil N\cdot p \rceil 
    &= \lceil (\lceil \frac{x+1}{1-p} \rceil - 1 + 1)\cdot p \rceil - 
        \lceil (\lceil \frac{x+1}{1-p} \rceil - 1) \cdot p \rceil \\
    &= \lceil \lceil \frac{(x+1)b}{b-a} \rceil \cdot \frac{a}{b} \rceil 
        - \lceil (\lceil \frac{(x+1)b}{b-a}\rceil - 1) \cdot \frac{a}{b} \rceil \\
    &= \lceil (m\cdot (b-a) - \frac{i}{b-a}) \cdot \frac{a}{b} \rceil 
        - \lceil (m\cdot (b-a) - \frac{i}{b-a} - 1) \cdot \frac{a}{b} \rceil 
        && \text{for some } 0 \leq i < b-a, i \in \mathbb{Z}; m \in \mathbb{Z} \\
    &= m\cdot (b-a) + \lceil - \frac{i}{b-a} \cdot \frac{a}{b} \rceil 
        - m\cdot (b-a) - \lceil (- \frac{i}{b-a} - 1) \cdot \frac{a}{b} \rceil 
        && \text{ since } m\cdot(b-a) \in \mathbb{Z} \\
    &= \lceil - \frac{i}{b-a} \cdot \frac{a}{b} \rceil 
       - \lceil (- \frac{i}{b-a} - 1) \cdot \frac{a}{b} \rceil \\
    &= 0 - \lceil (- \frac{i}{b-a} - 1) \cdot \frac{a}{b} \rceil 
       && \text{ since } 0 \leq \frac{i}{b-a} < 1; 0 < \frac{a}{b} < 1 \\
    \tag{A.4}

Now looking at the last line and looking at the expression in the ceil function,
we can see that:

.. math::

    (- \frac{i}{b-a} - 1) \cdot \frac{a}{b} & > -\frac{2a}{b} 
        && \text {since } \frac{i}{b-a} < 1 \\
    &\geq -1
        && \text {since } \frac{a}{b} \leq 0.5 \text{ using assumption } p=\frac{a}{b} < \frac{b-a}{b} = 1-p \\
    \tag{A.5}

So :math:`(- \frac{i}{b-a} - 1)\cdot \frac{a}{b} > -1` (and obviously :math:`< 0`), therefore
Equation A.4 resolves to :math:`- \lceil (- \frac{i}{b-a} - 1) \cdot \frac{a}{b} \rceil = 0` as required.

|h2| Notes |h2e|

.. [1] I actually liked most of the communication theory courses, it was interesting to learn the basics of how wired/wireless signals are transmitted and how they could be modelled using math.
    
