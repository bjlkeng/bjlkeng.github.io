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
commentary on my experience with everything.  This post is a lot less heavy on
the math as compared to my previous ones, but it still got some good substance
so do read on!

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
topics like fine-tuning or efficient inference  and the like.)

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

Large Language Models
---------------------

A `large language model (LLM) <https://en.wikipedia.org/wiki/Large_language_model>`__
is a `language model <https://en.wikipedia.org/wiki/Language_model>`__ that is... large.
First, a language model is simply a statistical model that tries to model:

.. math::

   P(w_m | w_{m-k}, \ldots, w_{m-1}) \tag{1}

In other words, given some context of previous words (although theoretically it can be surrounding words too)
:math:`w_{m-k}, \ldots, w_{m-1}`, try to predict the probability distribution for the next word :math:`w_m`.
Basically, the model predicts a probability for each possible next word.  Here word is not necessarily a word,
it can be a character, word or more commonly a `token <https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/tokens>`__.
Model in this case can be something simple like a `Markov chain <https://en.wikipedia.org/wiki/Markov_chain>`__, 
a `count based n-gram model <https://en.wikipedia.org/wiki/Word_n-gram_language_model#Approximation_method>`__,
or even a trillion parameter `transformer <https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>`__ neural network.
And finally "large" is a moving target without a precise definition.  
Nowadays, you probably need to have 1 billion `parameters <https://en.wikipedia.org/wiki/Statistical_parameter>`__
(or neural network weights) to be even be close.  For context 
`GPT-2 <https://en.wikipedia.org/wiki/GPT-2>`__ has 1.5B parameters, 
`GPT-3 <https://en.wikipedia.org/wiki/GPT-3>`__ has 175B parameters, and
the LLaMA has variants from 7B - 65B parameters.

In this post, I won't try to explain transformers in detail because I know I'm going to 
go too deep.  Instead, I'll refer you to these posts on `transformers <https://www.borealisai.com/research-blogs/tutorial-14-transformers-i-introduction/>`__, their `extensions <https://www.borealisai.com/research-blogs/tutorial-16-transformers-ii-extensions/>`__,
and their `training <https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/>`__ from Borealis
(where I currently work).  

If you aren't quite interested to go that deep, I'll give you the gist for our purposes.  
Transformers are a scalable neural network architecture that allows you to train
really high capacity (i.e., parameter) models.  The architecture accepts a sequence
of tokens represented as vectors as input, and in the "decoder" variant the
architecture can predict the next token after the input as in Equation 1.
Using various methods to select a specific next token, you append it to the
input, generate another token and so on until you generate a new sequence of
text.

The important part from this description is the original input you specify to
the LLM is called the **prompt**.  In `instruction tuned or aligned LLM models <https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/#Reinforcement_learning_from_human_feedback_RLHF>`__,
the prompt is essentially giving the LLM an instruction or query in natural
language, and it will iteratively (also called "auto regressively") generate
new text that (ideally) gives you a good response.  Unexpectedly, making
these LLM's really large and aligning them with human goals makes them
not only really good at understanding and writing natural language, but also
quite good at reasoning (debatable).  The prompt is critically important
to ensuring your LLM produces good output.  Instructing the LLM to "think
critically" or go "step by step" seems to produce better results, so subtle 
language cues can make a big different in the quality of output.

The other important part is the :math:`m` in Equation 1, which is also called the
**context window** length.  This is basically the size of "memory" the LLM has
to understand what you've input to it.  Modern commercial LLM's have context
windows in the thousands but some have context windows as long as 100K.  In the
basic case, LLM's will only perform well at context window lengths at or
below what it was trained on even the transformer architecture can mechanically
be extended to arbitrary lengths.

LLM's like many of its predecessor language models can also generate 
`embedding <https://en.wikipedia.org/wiki/Word_embedding>`__ from their input
prompts.  These are some combination of internal vectors that the underlying
transformer generates.  They map the input tokens to a new latent space that
typically will cluster similar concepts together, making them extremely useful
for downstream applications (see RAG below).

Lastly, due to the massive number of parameters, training these LLM's are
prohibitively expensive.  Training these 100+B models can be on the order
of millions of dollars (assuming you can even get a cluster of GPUs).
Inference on these models is relatively less compute intense but is more
limited by GPU VRAM, which usually still requires a distributed cluster.
Smaller models (e.g. 7B parameter) and advances in quantization and related
compression have inference (and sometimes training) running on single machines,
sometimes even without GPUs.

See `Borealis' post on LLMs <https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/#Reinforcement_learning_from_human_feedback_RLHF>`__, which is much more accessible than a lot of the
interweb posts out there.


Retrieval-Augmented Generation
------------------------------

`Retrieval-Augmented Generation (RAG)
<https://eugeneyan.com/writing/llm-patterns/#retrieval-augmented-generation-to-add-knowledge>`__
enhances a large language model by first retrieving relevant data and combining
it with the input to improve results.  This technique is typically used in
question and answering scenarios.  The name is fancier than it sounds (at least
for the main concept), LangChain has a good summary on its `Question Answering
Over Documents <https://docs.langchain.com/docs/use-cases/qa-docs>`__ page that
is roughly summarized below.

For the setup, you build an index of your documents representing each typically
as a word / sentence / paragraph `embedding <https://en.wikipedia.org/wiki/Word_embedding>`__ 
as follows:

1. Due to the limitations of LLMs, you will typically split your documents into
   bite-sized chunks that fit into the LLM's context window.
2. Create an embedding from each of your chunks.
3. Store documents in a vector store that can find the top-K matching
   chunks for a given embedding query.

Once you have a vector store, answering proceeds as follows:

1. Take the input question and convert it to an embedding.
2. Look up top-K relevant chunks in your vector store.
3. Construct a prompt based on the input question and these chunks.
4. Send the prompt to an LLM and return the result.

The original `RAG paper <https://arxiv.org/abs/2005.11401>`__ was written
before LLM's got really powerful so it seems that they do a bunch of other
fancy tricks.  However with LLM's, you don't need to seem to do much more than
the above to get pretty good results.  As far as I can tell, most setups will
do some variation of the above without much more effort.  As with most
LLM related things, the prompt is important (along with how many k documents to
include).  Similarly, the `chunking
<https://www.pinecone.io/learn/chunking-strategies/>`__ step may also be
important depending on your problem.

LLM Fine-Tuning
---------------

`Fine-tuning <https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)>`__ an
LLM is precisely the concept as it is used in other transfer learning
applications.  The main idea is to take an existing trained model ("pre-trained model"),
and modify the weights in order to adapt it to a different task.  The
modification of the weights can be for a subset of the layers, all of them,
or even none of them but effectively modifying the weights by augmenting
the model with additional trainable parameters.  Variants of the latter has
been a `popular technique <https://arxiv.org/abs/2106.09685>`__ to cheaply
fine-tune an existing LLM reducing the cost by orders of magnitude compared
to training the base model (or naively directly fine-tuning an LLM).  Typically
the fine-tuning uses a lower learning rate so you retain a substantial portion
of the learning of the pre-trained model.

The above "alignment" step is a form of fine-tuning where the base language
model is only good at predicting the next token, while fine-tuning gives it the
ability to follow instructions and respond as humans would expect.  Other
examples include training with more specific data for a task (e.g. Medical Q&A),
which has shown to improve performance over generic models.

OpenAI and Langchain APIs
-------------------------

Most of you will be familiar with `OpenAI <https://openai.com/>`__, most likely
from their breakout product `ChatGPT <https://chat.openai.com/>`__ that was probably
the first widespread demonstration of what LLM's could do (particularly because it
could follow instructions).  What's probably also obvious to most people is that
OpenAI has many `APIs <https://platform.openai.com/docs/introduction>`__ that
allow programmatic access to all the functionalities of ChatGPT and more.

The APIs are HTTP endpoints that have two officially released libraries in for
Python and Node.js (as well as other community maintained ones).  The most relevant
APIs related to this post are ones to call the via the `chat/completions` to respond
to a prompt, and the fine-tuning API to train a model on my own data.  The cost
is usually priced per 1000 tokens for both completion APIs and fine-tuning.
The latter charges different rates for training and inference depending on the
model.

For most of their language APIs, you can select which model you want to use.  The models
are roughly binned into how powerful each on is with the original ChatGPT using
`gpt-3.5-turbo` (with some details), `gpt-4` being their most capable ones, and others
being of the GPT-3 generation without instruction fine-tuning with various
model sizes (as I understand).

Working with the OpenAI APIs is pretty straight forward, but often times you want
additional functionality (such as RAG) and `Langchain <https://www.langchain.com/>`__
is one of the *many* libraries that fills in the gap.  It appears to be one of the
first and thus relatively popular at the moment, but things are changing fast.
Langchain has a Python library and a more recent JavaScript one, both of which
I used in this project.

The main advantage of Langchain (in my opinion) is that they have many predefined
patterns that you can put together such as RAG.  They have numerous examples
along with the building blocks you need to set up a default LLM application
with components such as predefined prompts, inclusion of various vector
databases, and integration with all popular LLM provider libraries.  It's hard to
say if this will be the LLM library of the future but it's definitely a useful
library to get up and running quickly.

Cloudflare Workers
------------------
`Cloudflare workers <https://workers.cloudflare.com/>`__ is a serverless code platform
developed by Cloudflare.  Although the large cloud providers (also known as
hyperscalers) generally have a serverless code offering (e.g. AWS Lambda), Cloudflare
touts several advantages such as:

* Automatic scaling 
* High performance
* Low latency startup time
* Better developer experience (DX)

One of the fundamental ideas is that you shouldn't have to think about the underlying
infrastructure at all, just deploy and have it work.  

Of course, these benefits do come with tradeoffs.  Their serverless code 
`runs in V8 isolates <https://developers.cloudflare.com/workers/learning/how-workers-works/>`__,
which is the same technology that Chrome's JavaScript engine uses to sandbox
each browser tab, which enables things such as the high performance and low
latency.  The obvious limitation here is that it only runs JavaScript.
While that is a big limitation, V8 also supports `WebAssembly <https://webassembly.org/>`__,
which opens the door to other languages such as Rust, C, Cobol (compiling to
WebAssembly). Other languages such as Python, Scala and Perl are enabled by
other projects that exist to make those languages work within a JavaScript
environment, often times with some reduced functionality (e.g. not all
libraries are available).

The other non-obvious thing is that although the Worker environment very
much behaves similar to Node.js, it is missing some key components due
to the security model that Cloudflare has implemented.  A glaringly obvious
limitation is that there is no filesystem.  This caused some trouble as I
mention below.

The other relatively large blocker, at least until recently, was that there was
no state management within the ecosystem.  You could make a call out to an
external database via an HTTP call, but the platform didn't natively support
it.  Cloudflare has been pushing hard on the innovation to make their solution
full stack by including things such as a zero-egress fee S3 compatible object store `R2 <https://www.cloudflare.com/developer-platform/r2/>`__, 
an eventually consistent key value store `Workers KV <https://www.cloudflare.com/developer-platform/workers-kv/>`__, 
a serverless SQL databse `D1 <https://developers.cloudflare.com/d1/>`__, and
a transaction store with `Durable Objects <https://developers.cloudflare.com/durable-objects/>`__.
Some of these are still in beta but Cloudflare's track record is pretty good at
building thoughtful additions to their platform with good DX.  It remains to be
seen if they can truly disrupt the established hyperscaler dominance.


ROUGE Metric
------------

The `ROUGE <https://en.wikipedia.org/wiki/ROUGE_(metric)>`__ or Recall-Oriented
Understudy for Gisting Evaluation is a family of metrics to evaluate
summarization and machine translation NLP tasks.  They work by comparing
the automatically generated proposed (hypothesis) text to one or more reference texts
(usually human generated).  Evaluation will depend very heavily on the meaning
of the text so (at least before the LLM revolution) it is desirable to use a
simple mechanical metric such as ROUGE that does not depend on the meaning.

ROUGE has many different variants with the simplest one called `ROUGE-N` being
based on the overlap of `N-grams <https://en.wikipedia.org/wiki/N-gram>`__
(word level) between the hypothesis text (:math:`s_{hyp}`) and reference text
(:math:`s_{ref}`) given by the formula:

.. math::

   ROUGE-N = \frac{\big| \text{N-GRAM}(s_{hyp}) \cap \text{N-GRAM}(s_{ref}) \big|}{\big|\text{N-GRAM}(s_{ref})\big|} \tag{2}

where :math:`\text{N-GRAM}(\cdot)` generates the multiset of (word-level) n-gram tokens and the
intersection operates on multisets.

Since we're using :math:`s_{ref}` in the denominator, it's a recall oriented
metric.  However, we could just as well use :math:`s_{hyp}` in the denominator
and it would be the symmetrical precision oriented metric.  Similarly, 
we could compute the related `F1-score <https://en.wikipedia.org/wiki/F-score>`__
with these two values.  This is the evaluation metric that I'll use later on
to give a rough idea of how good the LLM performed.

.. admonition:: Example 1: Calculating the ROUGE-2 score.

    Consider a hypothesis text summary and the reference text (I used GPT-4 to
    generate them both):

    .. math::
    
        s_{hyp} &= \text{"AI accelerators facilitate extensive text processing in large language models"} \\
        s_{ref} &= \text{"Large language models use AI accelerators for improved processing and training."} \\
        \tag{3}

    We can compute the multiset of n-grams (ignoring capitalization) and their intersection as:

    .. math::

        \text{1-GRAM}(s_{hyp}) &= [ai, accelerators, facilitate, extensive, text, processing, in, large, language, models] \\
        \text{1-GRAM}(s_{ref}) &= [large, language, models, use, ai, accelerators, for, improved, processing, and, training] \\
        \text{1-GRAM}(s_{hyp}) \cap \text{1-GRAM}(s_{ref}) &= [large, language, models, ai, accelerators, processing] \\
        \tag{4}

    We can then calculate the cardinality of each and finally compute the ROUGE-1 score:

    .. math::

        \big|\text{1-GRAM}(s_{hyp})\big| = 10,
        \big|\text{1-GRAM}(s_{ref})\big| = 11,
        \big|\text{1-GRAM}(s_{hyp}) \cap \text{1-GRAM}(s_{ref})\big| = 6 

    .. math::
        \text{ROUGE-1} = \frac{\big| \text{1-GRAM}(s_{hyp}) \cap \text{1-GRAM}(s_{ref}) \big|}{\big|\text{1-GRAM}(s_{ref})\big|}
         = \frac{6}{11} \approx 0.54 \\
         \tag{5}

    Similarly, the precision variant yields :math:`0.6` and the F1-score yields approximately :math:`0.57`.


Project Details
===============

This section gives an overview of the project components and highlights some of the details
that are not apparent from the code.  
All the `code is available <https://github.com/bjlkeng/bjlkengbot>`__ on Github
but please keep in mind that it's a one-off so I know it's a mess and don't
expect any reuse (besides the LLM related code will probably be out of date in
a few months anyways).

Crawler 
-------

The first thing I needed to do was gather a corpus of my writing.  Luckily,
there was a readily available corpus on my personal site `<https://www.briankeng.com>`__.
The posts have varying lengths, contain lots of quotes, and sometimes contain
dated information but generally I think my writing style hasn't changed too
much so I thought it would be interesting to see how it would do.  

I did the easiest thing I could to capture the text content and used the
`Scrapy <https://scrapy.org/>`__ library to crawl my site and captured the
title, URL and text content.
In total I crawled 173 pages (posts and a couple of selected pages) containing
my writing including the About Me page.

Next, the data was chunked into LLM-sized pieces.  Here I used the 
`RecursiveTextSplitter <https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter>`__.
This splitter is nice because it will try to group things by paragraphs, then
sentences, and then words, intuitively keeping semantically related pieces
together.  You can additionally utilize the OpenAI tokenizer using `from_tiktoken_encoder()`
to match the token counts that OpenAI's API expects.
A chunk size of 900 tokens with 100 overlapping tokens.  These numbers
were chosen because I was planning to send 4 documents into the RAG workflow so
I wanted it to be less than the default 4096 token window for the ChatGPT3
endpoint.

This was done as a preprocessing step because (as we will see later) the
Langchain JavaScript library doesn't (at the time of writing to my knowledge)
have the specific splitter + OpenAI tokenizer.  So I thought I would just split
the text into the appropriate chunks first and then not have to worry about
doing much manipulation in JavaScript.  The resulting output was a JSON file
containing an array of objects with the chunked text, and the associated
URL/title metadata for each chunk.

Embeddings and Vector DB
------------------------

- Doing the indexing in JS, saving the internal object as JSON (nice thing about JavaScript)
- Use OpenAI embedding endpoint
- MemoryVectorStore
- Langchain handles all of this for me pretty much
- Node allows read/write to disk but Cloudflare does not
    
Fine Tuning
-----------

Cloudflare Worker
-----------------

Webpage 
-------

Model Evaluation
================
n


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


Further Reading
===============

* `A High-Level Overview of Large Language Models <https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/>`__
* `Building LLM-based Systems & Products <https://eugeneyan.com/writing/llm-patterns/#retrieval-augmented-generation-to-add-knowledge>`__


.. [1] In fact, there are several projects going on at work that are related to this topic but since I'm in a technical management role, I spend almost no time coding or directly doing research.  Thus, this blog is my outlet to satisfy my curiousity both also help with staying current on both fronts.
