.. title: Functional Programming: A first step
.. slug: functional-programming-a-first-step
.. date: 2015-08-02 22:03:05 UTC-04:00
.. tags: functional programming, James Hague
.. category: 
.. link: 
.. description: A first step into functional programming.
.. type: text

I have something to admit: I've never done any serious programming
in a functional programming language.  Yes, yes, I've done some small school
assignments in `Scheme <https://en.wikipedia.org/wiki/Scheme_%28programming_language%29>`_ (a dialect of Lisp), 
even helping out my friend from another university with his Scheme assignment
but nothing *real*.  Does this make me a worse programmer?  Probably not, I
imagine most developers haven't done anything *real* with functional
programming.  Although that's probably not what you'd expect from reading
`Hacker News <https://news.ycombinator.com/>`_, where you don't know
programming if you haven't tried Clojure or Haskell.  My position is much more
pragmatic: I'm interested in tools and techniques that help me solve problems
faster, cleaner and with less headache.  Functional programming is *supposed*
to help with at least some of that.

James Hague has an interesting opinion on this subject in his not-so-subtle
post titled `Functional Programming Doesn't Work (and what to do about it)
<http://prog21.dadgum.com/54.html>`_.  In his `follow-up
<http://prog21.dadgum.com/55.html>`_, he hit the nail on the head with this:

    My real position is this: 100% pure functional programing doesn't work. Even
    98% pure functional programming doesn't work. But if the slider between
    functional purity and 1980s BASIC-style imperative messiness is kicked down a
    few notches--say to 85%--then it really does work. You get all the advantages
    of functional programming, but without the extreme mental effort and
    unmaintainability that increases as you get closer and closer to perfectly
    pure.





.. code:: python

 def my_function():
     "just a test"
     print 8/2
