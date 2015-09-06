.. title: A First Post
.. slug: a-first-post
.. date: 2015-08-02 5:03:25 UTC-04:00
.. tags: first post, Github, Nikola, satisficing, James Hague
.. category: 
.. link: 
.. description: A First Post
.. type: text

.. |br| raw:: html

   <br />

Hi all, this the first post in my technical blog.  For a long time I've resisted making
a blog about anything technical because, to be frank, I didn't feel like I
was qualified.  What makes me more qualified now?  Nothing really, I'm just a bit less
afraid of `being wrong <https://xkcd.com/386/>`_.

So maybe a good place to start is what I used to setup this blog and what the
domain/title mean.  
For my personal blog (`www.briankeng.com <http://www.briankeng.com/>`_), where
I discuss random worldly wisdom that I come across, I use `WordPress <https://wordpress.org/>`_.  
For this one, I used something a bit more technical (because why not?): 

`Github Pages <https://pages.github.com/>`_ 
  Great place to host a static website (free) but also has some of that "hacker
  cred".  It's quite easy to use (if you've used git before) and also quite
  well `documented <https://help.github.com/articles/setting-up-a-custom-domain-with-github-pages/>`_
  on how to get your domain setup.

|br| `Nikola <https://getnikola.com/>`_ (static site generator)
  The recommended static site generator with Github Pages is Jekyll but one
  thing I wanted for this site is to make use of `IPython Notebooks
  <http://ipython.org/notebook.html>`_.  After a bit of searching, I found
  Nikola to be a good match because it supports IPython Notebooks out of the box
  (as opposed to Jekyll which required some fiddling), and seemd to have a reasonably
  simple setup.  The really interesting line from the `documentation <https://getnikola.com/handbook.html#getting-more-themes>`_
  is this quote: ``DON'T READ THIS MANUAL. IF YOU NEED TO READ IT I FAILED, JUST USE THE THING.``
  Exactly the sentiment I was going for.

|br| `Bootstrap3 <https://themes.getnikola.com/#bootstrap3>`_ theme and `Readable <bhttp://bootswatch.com/readable/>`_ Bootstrap theme.
  Bootstrap3 is the default theme for Nikola, conveniently using `Bootstrap <http://getbootstrap.com/>`_, 
  the most popular HTML/CSS/JS framework for the web.  Since I'm not a very
  good designer using high quality themes/framework is definitely ideal.  
  
  The default Nikola theme, however, was not too my liking.  A bit too spartan for me.  Fortunately, Nikola has a really cool
  feature where you can take an existing Bootstrap theme from `http://bootswatch.com <http://bootswatch.com>`_ 
  and skin the default Nikola Bootstrap theme.  I chose the `Readable <bhttp://bootswatch.com/readable/>`_ theme because I'm
  a huge fan of making things easy-to-read (you should see how large the font
  size is on my devices, frequently referred to as "old people font size").

  Even with the new Bootstrap theme, the site layout didn't quite look right.
  So I fiddled around with the layout and placement of items, pretty much
  copying the layout of another blog I really enjoy `programming in the twenty-first century <http://prog21.dadgum.com/>`_ by James Hague.
  It was interesting playing around with HTML templates and CSS but pretty much
  confirmed that it's not really what I like doing.  Getting to a satisfactory
  layout was as much effort as I really wanted to put in.

.. TEASER_END

|br|

As for the name of the blog, ever since I read the autobiography of `Herbert Simon <https://en.wikipedia.org/wiki/Herbert_A._Simon>`_,
I've been fascinated by his idea of `bounded rationality <https://en.wikipedia.org/wiki/Bounded_rationality>`_ which is the idea that:

    "when individuals make decisions, their rationality is limited
    by the information they have, the cognitive limitations of their minds, and
    the time available to make the decision."

Along with this idea, Simon also gives the idea of *satisficing*, a term
combining the meaning *satisfy* and *suffice*, a stark contrast to optimal
decision making.  Not only was he quite ahead of the times applying this idea
of bounded rationality to economics/social science analysis, it also really
speaks to his strong inclination to look at reality in the most rational way.
This is as opposed to how many academics behave by trying to fit reality
into their elegant mathematical models (A pithy quote from Galbraith:
``economists are most economical about ideas. They make the ones they learned
in graduate school last a lifetime``).  

Simon is definitely a personal hero of mine, a true polymath.  This is quite
apparent from Simon winning both a Nobel Prize *and* a Turing award (the Turing
award was for his contributions to artificial intelligence and IPL, a precursor
to Lisp).  Beyond his massive contributions to the social and computer
sciences, his multi-disciplinary approach to problems along with a strong
tendency towards rational thinking really speak to me.  So naturally, I
don't mind lifting the terms *satisficing* and *bounded rationality* to use
as names for the domain/blog title.

In this weblog, I hope to apply the spirit of these two ideas, trying to
understand and explain technical subjects without getting bogged down in the
minutiae that so frequently surrounds technical subjects.  Instead, try to
focus on the ideas that actually help us move forward in solving practical
problems to a satisfactory degree (satisficing).

With respect to the subject area of the blog, it will probably vary quite a
bit.  There will probably be a lot of focus on programming and data science
since that's what I spend most of my time at work doing.  However, don't be
surprised if I write about other various technical subjects that interest me
such as math or investing.  Real world problems rarely fit into such neat
classifications, so why limit what I write about?  Hope you enjoy it.

