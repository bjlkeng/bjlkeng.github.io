.. title: Building a Q&A Bot of Myself
.. slug: building-a-qa-bot-of-me-with-openai-and-cloudflare
.. date: 2023-07-28 20:56:42 UTC-04:00
.. tags: mathjax
.. category: 
.. link: 
.. description: 
.. type: text

Unless you've been living under a rock, you've probably heard of large language
models (LLMs) such as ChatGPT or Bard.  I'm not one for riding a hype train but
I do think LLMs are here to stay and either are going to have an impact as big 
as mobile as a platform (my current best guess) or perhaps something as big as 
the Internet itself.  In either case, it behooves me to do a bit more
investigation on this popular trend both from an application and 
coding assistant point of view [1]_.  At the same time, there are a bunch
of other developer technologies that I've been wondering about like serverless computing
modern dev tools, and LLM-based code assistants, so I thought why not kill
multiple birds with one stone.

This post is going to how I built a question and answering bot of myself using
LLMs and my experience of the whole process using some modern developer tools
`ChatGPT <https://chat.openai.com>`__, `Github Copilot
<https://github.com/features/copilot>`__, `Cloudflare workers
<https://workers.cloudflare.com/>`__, and a couple of other related ones.
I start out with *my motivation* for doing this project, some brief background
on the technologies, a description of how I built everything, and finally some
commentary on my experience with everything.  If that interests you, read on!

*Note: This post is quite different from my previous ones with almost no
explanation of the math or techniques underneath, so if you came here for
that you may want to skip the post.  However, you might be interested
in the commentary at the end, which I think will be insightful to many.*

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

My Motivation: Why build an LLM Me?
===================================

As I mentioned above, *most* of the reason why I did this little project was to
get better intuition with LLMs.  Practically that meant on the application level
and not training foundation models myself.  Realistically only a few
organizations are setup to do the really large training, and I'm (a) not in one
of those companies and (b) likely wouldn't be working on that project given my
experience.  So the result is that I needed to learn more about how to use LLMs
in application.  (Although this does not preclude me from exploring other technical
topics like fine tuning or efficient inference  and the like.)

The other LLM-related reason I did this project was to play around with modern
development tools.  Github Copilot and ChatGPT (at least from the outside) 
had the potential to be a step change in productivity so I would be irresponsible
not to learn more about them.  This is even true if I'm not actually coding on
a daily basis because it will help me understand how it could affect my teams
(which do a lot of coding).

Github Copilot is easiest to setup within `VSCode
<https://code.visualstudio.com/>`__ so I decided to take that for a spin.
Coming from Vim + Jupyter notebooks (depending on the task) for the past 15
years or so, it was probably time to try out a new IDE.  Everything has Vim
bindings nowadays (including Jupyter), and I mostly just use the standard
commands.  The appealing thing about VSCode is the ability to manage Jupyter
notebooks alongside my code files, which I always found super annoying switching 
back and forth between.

The last piece of technology I wanted to play with was Cloudflare
workers and the related ecosystem of infrastructure.  To be honest, the main
reason for wanting to learn more about this is primarily because I recently
invested in the stock and I wanted to learn more about their "Act 3 products",
which is primarily their serverless development platform.  Serverless always
seemed interesting but limited to a subset of use-cases so I never gave it 
much attention except playing around with AWS lambda early on.  Cloudflare
has a very opinionated way of doing things with a unique architecture so
it was definitely an interesting experience.

Finally, the scope of the project needed to be sufficiently small that I 
wouldn't spend too much time on it but at the same time actually get enough
experience test driving the technologies above.  Most importantly though,
it should be fun!  So it was only natural to stroke my own ego and try
to make a virtual version of myself, which checked all the boxes.  Despite it
following almost exactly the same idea from popular sci-fi `making killer
robots from their online presence <https://en.wikipedia.org/wiki/Caprica>`__, I was not
worried at all because (a) I have very little training data (my online presence
is small and explicit on purpose), and (b) I'm highly doubtful that LLMs are
that powerful.  In any case, enjoy the writeup!

Background
==========

LLM Question and Answering
--------------------------

LLM Fine-Tuning
---------------

OpenAI and Langchain APIs
-------------------------

Cloudflare Workers
------------------

Rouge Metric
------------

Project Details
===============

Crawler 
-------

Embeddings and Vector DB
------------------------
    
Fine Tuning
-----------

Model Worker
------------

Webpage 
-------

Model Evaluation
================



Commentary
==========

LLMs as Coding Assistants
-------------------------

* Github
* ChatGPT
* OpenAI API

Langchain
---------

VSCode vs. Vim
--------------

Cloudflare
----------

* Worker
* Email worker
* DDos protection
* Domain registration

LLMs: Do we need to worry?
--------------------------

Conclusion
==========

It was a fun project and I might end up doing more of them instead of diving
deep into the math and algorithms.

.. [1] In fact, there are several projects going on at work that are related to this topic but since I'm in a technical management role, I spend almost no time coding or directly doing research.  Thus, this blog is my outlet to satisfy my curiousity both also help with staying current on both fronts.
