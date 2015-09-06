.. title: Functions and Procedures
.. slug: functions-and-procedures
.. date: 2015-09-06 03:04:05 UTC-04:00
.. tags: procedures, functions, functional programming, James Hague, Greg Ward, fundamentals
.. category: 
.. link: 
.. description: A look at two basic types of subroutines.
.. type: text

.. |br| raw:: html

   <br />

.. |H2| raw:: html

   <h3>

.. |H2e| raw:: html

   </h3>

I have something to admit: I've never done any serious programming in a
functional language.  Yes, yes, I've done some small school assignments in
`Scheme <https://en.wikipedia.org/wiki/Scheme_%28programming_language%29>`_ (a
dialect of Lisp), even helping out my friend from another university with his
Scheme assignment but nothing *real* [1]_.  Does this
make me a worse programmer?  Probably not, I imagine most developers haven't
done anything *real* with functional programming.  Although that's probably not
what you'd expect from reading `Hacker News <https://news.ycombinator.com/>`_,
where you don't know programming if you haven't tried Clojure or Haskell [2]_.

My position is much more pragmatic: I'm interested in tools and techniques that
help me solve problems faster, cleaner and with less headache.  Ideas and
concepts from functional programming languages are *supposed* to help with at
least some of that -- and they do; they're just not a panacea for all your
programming woes.  Like most things a solid grasp of the fundamentals goes a
long way.  So the topic of this post is about something pretty fundamental: a
subroutine.  In particular, two important kinds of subroutines:
procedures and pure functions.

.. TEASER_END

|h2| Programming is Organization |h2e|

Let me digress for a moment because I want to discuss an incredibly important idea
that James Hague discusses in his post `Organizational Skills Beat Algorithmic
Wizardry <http://prog21.dadgum.com/177.html>`_.  He nails down one of the most
important points about software development: organization.  (emphasis mine)

    When it comes to writing code, the number one most important skill is how to
    keep a tangle of features from collapsing under the weight of its own
    complexity... there's always lots of state to keep track of, rearranging of values,
    handling special cases, and carefully working out how all the pieces of a
    system interact. To a great extent **the act of coding is one of organization**.
    Refactoring. Simplifying. Figuring out how to remove extraneous manipulations
    here and there.

As much as it is fun debating the merits of one language/framework/technology
versus another, it's much more practical to talk about ways we can organize
programs in a more efficient manner (in whatever language/framework/technology
we're currently using).  Now back to the main show...

|h2| A Function By Any Other Name |h2e|

It's funny that one of the first things we learn when programming is the concept
of a subroutine: *a set of instructions designed to perform a frequently used
operation within a program*, which is supposed to help organize your program.
Great, but I don't recall learning much about the different kinds of
subroutines or even really best practices for using them.  For some reason
that's just something you have to figure out yourself.  Let's try to be a bit
more explicit.

In my mind, a useful (but perhaps not universally accepted) classification of
subroutines breaks them down into two general categories: procedures and pure
functions.

  * A `procedure` is a sequence of commands to be executed.  These are
    usually used for `doing stuff`.  Typically, these will involve side-effects
    (such as changing the state of variables, outputting to the screen, or
    saving things to a file etc.).  Procedures don't have return values.
  * A (pure) `function` computes a value (and returns it).  These are for
    `computing stuff`.  Just like a function in math, for the same set of
    inputs, it will always return the exact same output.  Functions don't have
    side-effects.

Notice that these are not the only two ways to think about subroutines.  There
is a type of subroutine that returns something *and* has side-effects (among
others).  But I argue that these two are the most constructive ways to think
about subroutines.

The most popular languages out there today don't really make a distinction
between these two types of subroutines, but that doesn't mean you shouldn't!
The reason to look at subroutines this way is because of a general rule of
thumb that I came across by Greg Ward in his talk at Pycon 2015,
`How to Write Reusable Code <https://www.youtube.com/watch?v=r9cnHO15YgU>`_
(`slides
<)https://github.com/PyCon/2015-slides/tree/master/Greg%20Ward%20-%20How%20to%20Write%20Reusable%20Code>`_):

    Every [subroutine] should either return a value or have a side effect: never both.

This is great rule of thumb that's hard to appreciate until you've made the mistake of 
violating it and have it come back to bite you in the arse.  Greg goes on to give
a couple of great examples (from actual code reviews he has done).  Here's 
one of his examples where this rule of thumb is violated:

.. code:: python
 
 def get_foo(self, foo=None):
     '''Query the server for foo values.
     Return a dict mapping hostname to foo value.
 
     foo must be a dict or None; if supplied, the
     foo values will additionally be stored there
     by hostname, and foo will be returned instead
     of a new dict.
     '''

Gee, I'm already confused even after reading the documentation (don't even get
me started on the mismatch with the function name).  Remember, we want to
build systems that don't "`collapse under the weight of its own complexity`" by
"`Simplifying. Figuring out how to remove extraneous manipulations here and
there.`"  Sure, giving it a second read, we can probably figure out what it does
but the fact that we need to think twice about it sure isn't helping the
complexity.  Imagine if every subroutine you wrote had this issue -- I don't envy
that code reviewer.

Greg goes on to give a better way to implement ``get_foo()`` as a pure function:

.. code:: python

 def get_foo(self):
     '''Query the server for foo values.
     Return a dict mapping hostname to foo value.
     '''

Much simpler and easy to understand: query the server, get back a ``dict``.
No extraneous mental overhead with the ``foo`` parameter.  It
may only be a small improvement but when building a large system, these small
things add up quickly (especially since complexity is likely multiplicative).


There's also this example involving C:
  
.. code:: c
 
 /**
  * Replace all 'e' characters in str with 'E'. Return the number of
  * characters replaced.
  */
 int strmunge_v1(string str) {
     ...
 }

He points out that this type of subroutine is pervasive in C and notes that the
only valid reason for violating this rule is for performance (which is probably
why you're programming in C in the first place!).  For the rest of us who aren't
writing performance critical code (come guys, that's most of you), a much
cleaner solution is not to have the side-effect and convert it to a pure
function:

.. code:: c

 /**
  * Return newstr, a copy of str with all 'e' characters replaced
  * by 'E', and nreplaced, the number of characters replaced.
  * (Assume language with multiple return values)
  */
 (string, int) strmunge_v2(string str) {
     ...
 }

The pure function has many benefits over the side-effect-ridden one (functional
programmers rejoice!) with the main one that it's easier to reason about: you
can look at the function in isolation of the entire program.  Write it
separately, review it separately, unit test it separately.  And once you're
convinced it works properly, you don't need to look at it again!  You can now
"abstract" that function out when reading the parent functions.  Awesome!
I'm a huge fan of making things `simpler <http://www.briankeng.com/about/>`_.


|h2| Fundamentals |h2e|

The reason that I decided to write this post is that lately, I've been using a
"procedure of (pure) functions" type pattern in my code.  My main logic
typically is some kind of procedure that farms out much of the work to pure
functions rather than mixing them (kicking my old performance-driven C++
mindset).  I find that it's been a very useful way to structure my programs and
generally just more pleasant to read.

After noticing this subtle shift in my code (and after watching Greg's talk), I
rediscovered my appreciation for the fundamentals.  I get the feeling that when
people want to learn something they conflate the most advanced ideas with the
most important.  There's definitely something to be said of taking a step back
and learning the fundamentals well.  Programming is no different in this
respect.  If you want to become strong at programming, start with the
fundamentals.

|br|
|br|

.. [1] "Real" programming work is kind of a vague word.  The way I'm using it here is any kind of sizable project, solving a non-trivial problem.  Most of the time these types of projects aren't measured in days or weeks  but rather months and years.

.. [2] Of course, I'm not saying Clojure and Haskell are bad languages or incapable of solving "real" problems with -- I'm almost positive they are fine languages to use.  I'm more of the opinion that, practically, it's harder to use them to solve many of the problems out there.  It's not just the issue from learning the FP conceptual point of view but also the fact that it's not that easy to find libraries, examples or even jobs that use these languages (although obviously some do exist).  Without a good "support structure" (including monetary compensation), it's hard to justify using a functional language.
