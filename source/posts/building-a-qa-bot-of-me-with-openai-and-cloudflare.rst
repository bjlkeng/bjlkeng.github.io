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
`ChatGPT <https://chat.openai.com>`__, `Github copilot
<https://github.com/features/copilot>`__, `Cloudflare workers
<https://workers.cloudflare.com/>`__, and a couple of other related ones.
I start out with *my motivation* for doing this project, some brief background
on the technologies, a description of how I built everything, and finally some
commentary on my experience with everything.

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

Why an LLM Me?
==============

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


.. [1] In fact, there are several projects going on at work that are related to this topic but since I'm in a technical management role, I spend almost no time coding or directly doing research.  Thus, this blog is my outlet to satisfy my curiousity both also help with staying current on both fronts.
