.. title: Iterative Summarization using LLMs
.. slug: iterative-summarization-using-llms
.. date: 2024-05-26 21:21:43 UTC-04:00
.. tags: LLM, summarization, fixed point, OpenAI
.. category: 
.. link: 
.. description: A short post on showing what happens if you keep summarizing a piece of text.
.. type: text

After being busy for the first part of this year, I finally have a bit of time
to work on this blog.  After a lot of thinking about how to best fit working on
this blog into my schedule, I've decided to *attempt* to write shorter posts.
Although I do get a lot of satisfaction writing long posts, it's not practical
because of the time commitment.  Better to break it up into smaller parts to be
able to jump around instead of being "stuck" writing one long post that I never
finish.  This also allows me to experiment with smaller scoped topics, which
hopefully will keep more more motivated to post.  Speaking of which...

This post is about answering a random thought I had the other day: what would
happen if I kept passing an LLM's output back to itself (sort of similar to
agents talking to each other).  I run a few experiments of trying to get the
LLM to iteratively summarize or rephrase a piece of text and the results are...
pretty much what you would expect.  But if you don't know what to expect, then
read on and find out what happened!

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


Setup
=====

The setup for experiments is really simple.  Start with one of two pieces of
data: A truncated (to fit within the 128k context) JSON of all my blog posts
from my `personal site <www.briankeng.com>`__ or an LLM generated list of 100
random facts.  Next, run an LLM asking it to either:
:code:`Summarize the following: {data}` or :code:`Rephase the following: {data}`.

And took the response from the LLM and fed it back 50 times.  I also used this
system prompt to help not lose too much data:

    The following is a conversation with an AI assistant. The assistant is helpful,
    creative, clever, and very friendly. The assistant will attempt to give a
    response that is concise but ensures that all the key points are included when
    relevant.

I ran combinations of the above using :code:`gpt-4o` and :code:`gpt-3.5-turbo` (using the
former to do the initial summary of the blog posts because of the context
window), and temperatures of :code:`0.0, 0.5, 1.0`.  That's it!
You can find my hacky code on `Github <https://github.com/bjlkeng/sandbox/tree/master/llm_fixed_point>`__
for these experiments.  Let's take a look to see what happened.

Experiments
===========

Blog Data
---------

First up is the summary of the blog post experiments shown in Figure 1 (nevermind the 42
in the label, that's just the random seed) where the X-axis is the iteration
number and the Y-axis is the length of the response.  The input data to this is
a summary by :code:`gpt-4o` of a scrape of my blog, which you can look at
`here <https://github.com/bjlkeng/sandbox/blob/master/llm_fixed_point/blog_gpt4o_summary.txt>`__.
It's about 7.6 KB in size compared to all my blog posts which are closer to
500KB.  You can see from the start we drop a lot of bytes cutting the size down
by more than half in all cases.

.. figure:: /images/llm_fixed_point_blog.png
  :height: 350px
  :alt: Blog post experiments
  :align: center

  **Figure 1: Length of response from LLM's for iterative summarization across {gpt-4o, gpt-3.5-turbo} and temperatures of {0.0, 0.5, 1.0}.**

What seems obvious is that most of the LLM's converged to a pretty short
response.  Here's an example from :code:`gpt-4o_42_0.5`:

    The author, an AI and data science expert affiliated with BorealisAI and
    the University of Toronto, blends their professional knowledge with
    interests in sports, music, and technology to offer varied reading
    suggestions. They are a respected speaker, patent holder, and researcher
    whose reflections from 2020-2022 emphasize family, career growth, humility,
    and mentorship. At Rubikloud, they prioritized people and learning,
    contributing to its acquisition by Kinaxis. Throughout the pandemic, they
    maintained routines and optimized their work environment. Their 2019 review
    highlights professional achievements, personal health, team management, and
    includes an inspiring story about Mozart.

I mean as a summary of my entire blog in less than 1000 bytes, it's not
terrible.  But the original summary, which was already greatly condensed from
all my writing, is 7x the size.  The only outlier is :code:`gpt-4o` at
:code:`temperature=0.0`, here's the summary it gave:

    The author is a Research Director at BorealisAI and an Adjunct Professor at the University of Toronto, specializing in Electrical and Computer Engineering. He enjoys sports, music, technology, and economics, and values simplicity and flexibility. His expertise includes AI for hardware design automation and teaching data structures and algorithms. He recommends "Outliers" by Malcolm Gladwell and "The Intelligent Investor" by Benjamin Graham.
    
    His contributions to data science and AI include talks, patents, and papers, focusing on machine learning, decision automation, and AI in retail, with notable patents in user behavioral segmentation and influencer analysis.
    
    In his yearly reviews, he reflects on professional and personal milestones:

    - **2022**: A stable year with family time and professional highlights, emphasizing mindfulness and quality time.

    - **2021**: Adaptation to the pandemic, starting at BorealisAI, and hopes for better social connections and fitness.

    - **2020**: Significant changes like his daughter's birth, buying a house, and Rubikloud's acquisition by Kinaxis, with challenges in maintaining hobbies and fitness.
    
    Reflecting on six years at Rubikloud, he emphasizes the importance of people, learning, and resilience, with the acquisition by Kinaxis marking a new chapter. Lessons from the pandemic include adjusting routines, managing stress, and improving remote work efficiency.
    
    The 2019 review covers professional highs and emotional lows, contributions to Rubikloud's growth, and challenges in work-life balance and burnout. Teaching was fulfilling, and personal hobbies progressed, with aims to improve health and fitness in 2020.
    
    He reviews books like "Tuesdays with Morrie" and "Radical Candor," offering insights on cherishing life, honest feedback, and learning strategies. A humorous story about Mozart highlights self-sufficiency and the importance of self-motivation and independence.

It's a respectable 2000 characters long and still maintains some of the
original format.  Perhaps what is most interesting here is that it looks like
that :code:`gpt-4o` reached a fixed point in this configuration.  The plateau in
Figure 1 in fact did produce the same summary over and over again.  This obviously
could only really happen at temperature 0 where each run is (mostly) deterministic.
:code:`gpt-3.5-turbo` showed similar behavior with a shorter summary at this
temperature but it did drop a few words at iteration 14 and again at iteration
49. This is likely because of the slight non-determinism of how OpenAI runs
the models even with a fixed random seed.

At any other larger temperature, we would expect deviation which explains all
the wiggles on the other runs.  We also see that the other :code:`gpt-4o` runs
with larger temperature did have similar length summaries in the first few
iterations but quickly devolved into much shorter ones where I assume it could
not make more concise.  The :code:`gpt-3.5-turbo` run's response length drops
pretty quickly after the first iteration though.

All these results aren't too surprising except for the "near" fixed point that
we saw with `gpt-4o`, temperature 0.  It just seems unlikely to me that it
would reproduce the exact same text instead of modifying a word here or there.
These LLM's are mysterious blackboxes indeed.

-----------------


Discussion
==========

* Temperature at 2.0 kept getting error:

    Error code: 500 - {'error': {'message': 'Failed to create completion as the model generated invalid Unicode output. Unfortunately, this can happen in rare situations. Consider reviewing your prompt or reducing the temperature of your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists. (Please include the request ID req_9e5be7d84b972df525ab75827f1ccc37 in your message.)', 'type': None, 'param': None, 'code': None}}

* Weights & Biases is quite good! Easy to use, some learning curve.  Obvious lock in problems.  Pretty expensive from an enteprirse point of view.
