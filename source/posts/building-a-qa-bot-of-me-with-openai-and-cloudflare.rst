.. title: LLM Fun: Building a Q&A Bot of Myself
.. slug: building-a-qa-bot-of-me-with-openai-and-cloudflare
.. date: 2023-09-24 20:56:42 UTC-04:00
.. tags: large language models, LLM, GPT, OpenAI, Cloudflare, Javascript, Q&A, LangChain, mathjax
.. category: 
.. link: 
.. description: 
.. type: text

Unless you've been living under a rock, you've probably heard of large language
models (LLM) such as ChatGPT or Bard.  I'm not one for riding a hype train but
I do think LLMs are here to stay and either are going to have an impact as big 
as mobile as an interface (my current best guess) or perhaps something as big as 
the Internet itself.  In either case, it behooves me to do a bit more
investigation into this popular trend [1]_.  At the same time, there are a bunch
of other developer technologies that I've been wondering about like serverless
computing, modern dev tools, and LLM-based code assistants, so I thought why not
kill multiple birds with one stone.

This post is going to describe how I built a question and answering bot of myself using
LLMs as well as my experience using the relevant developer tools such as
`ChatGPT <https://chat.openai.com>`__, `Github Copilot
<https://github.com/features/copilot>`__, `Cloudflare workers
<https://workers.cloudflare.com/>`__, and a couple of other related ones.
I start out with my motivation for doing this project, some brief background
on the technologies, a description of how I built everything including some
evaluation on LLM outputs, and finally some commentary.  This post is a lot
less heavy on the math as compared to my previous ones but it still has some
good stuff so read on!

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

As I mentioned above, *most* of why I did this little project was to
get a better intuition of LLMs.  Practically that meant on the application level
and not training foundation models myself.  Realistically only a few
organizations are setup to train really large models, and I'm (a) not in one
of those companies, and (b) likely wouldn't be working on that project given my
experience.  So the result is that I needed to learn more about how to use LLMs
in applications rather then train them from scratch (although this does not
preclude me from exploring other technical topics like fine-tuning or efficient
inference.)

The other LLM-related reason I did this project was to play around with modern
development tools.  Github Copilot and ChatGPT (at least from the outside) 
have the potential to be a step change in productivity so I would be irresponsible
not to learn more about them.  This is even true even if I'm not actually coding on
a daily basis because it will help me understand how it could affect my teams
(which do a lot of coding).

Github Copilot is easiest to setup within `VSCode
<https://code.visualstudio.com/>`__ so I decided to take that for a spin too.
Coming from Vim + Jupyter notebooks (depending on the task) for the past 15
years or so, it was probably time to try out a new IDE.  Everything has Vim
bindings nowadays (including Jupyter), and I mostly just use the standard
commands anyways.  The appealing thing about VSCode is the ability to manage
Jupyter notebooks alongside my code files, which solved an annoyance I always
had switching back and forth between tools.

The last piece of technology I wanted to play with was Cloudflare
Workers and the related developer platform.  To be honest, the reason for this
interest is that I recently invested in the stock and I wanted to learn more about their
"Act 3 products", which is primarily their serverless development platform.
Serverless always seemed interesting but limited to a subset of use-cases so I
never gave it much attention except playing around with AWS lambda early on.
Cloudflare has a very opinionated way of doing things with a unique
architecture so it was definitely an interesting experience.

Finally, the scope of the project needed to be sufficiently small that I 
wouldn't spend too much time on it but at the same time actually get enough
experience test driving the technologies above.  Most importantly though,
it should be fun!  So it was only natural to stroke my own ego and try
to make a virtual version of myself, which checked all the boxes.  Despite it
following almost exactly the same idea from popular sci-fi (i.e., `making killer
robots from their online presence <https://en.wikipedia.org/wiki/Caprica>`__), I was not
worried at all because (a) I have very little training data (my online presence
is small and very explicitly narrow), and (b) I'm highly doubtful that LLMs are
that powerful.  In any case, enjoy the write up!

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
Nowadays, you probably need to have at least a billion `parameters <https://en.wikipedia.org/wiki/Statistical_parameter>`__
(or neural network weights) to be even considered large.  For context 
`GPT-2 <https://en.wikipedia.org/wiki/GPT-2>`__ has 1.5B parameters, 
`GPT-3 <https://en.wikipedia.org/wiki/GPT-3>`__ has 175B parameters, and
the LLaMA has variants from 7B - 65B parameters.

In this post, I won't try to explain transformers in detail because I know I'm going to 
go too deep.  Instead, I'll refer you to these posts on `transformers <https://www.borealisai.com/research-blogs/tutorial-14-transformers-i-introduction/>`__, their `extensions <https://www.borealisai.com/research-blogs/tutorial-16-transformers-ii-extensions/>`__,
and their `training <https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/>`__ from Borealis AI
(where I currently work).  

If you aren't quite interested to go that deep, I'll give you the gist for our purposes.  
Transformers are a scalable neural network architecture that allows you to train
really high capacity (i.e., parameter) models.  The architecture accepts as input a sequence
of tokens represented as vectors, and the "decoder" variant of the
architecture can predict the next token after the input as in Equation 1.
Using various methods to select a specific next token, you append it to the
input, generate another token and so on until you generate a new sequence of,
for example, text.

The important part from this description is the original input you specify to
the LLM, which is called the **prompt**.  In `instruction tuned or aligned LLM models <https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/#Reinforcement_learning_from_human_feedback_RLHF>`__,
the prompt is essentially giving the LLM an instruction or query in natural
language (e.g., English), and it will iteratively (also called "auto regressive") generate
new text that (ideally) gives you a good response to your instruction.
Unexpectedly, making these LLMs really large and aligning them with human
goals makes them not only really good at understanding and writing natural
language, but also quite good at reasoning (debatable).  The prompt is
critically important to ensuring your LLM produces good output.  Instructing
the LLM to "think critically" or go "step by step" seems to produce better
results, so subtle language cues can make a big different in the quality of
the output.

The other important part is the :math:`m` in Equation 1, which is also called the
**context window** length.  This is basically the size of "memory" the LLM has
to understand what you've input to it.  Modern commercial LLMs have context
windows in the thousands of tokens but some have context windows as long as
100K.  In the typical case, LLMs will only perform well at context window
lengths at or below what it was trained on even though the transformer
architecture can mechanically be extended to arbitrary lengths.

LLMs like many of its predecessor language models can also generate 
`embedding <https://en.wikipedia.org/wiki/Word_embedding>`__ from their input
prompts.  These are some combination of internal vectors that the underlying
transformer generates.  They map the input tokens to a new latent space that
typically will cluster similar concepts together, making them extremely useful
for downstream applications (see RAG below).

Lastly, due to the massive number of parameters, training these LLMs are
prohibitively expensive.  Training these 100+B parameter models can be on the order
of millions of dollars (assuming you can even get a cluster of GPUs nowadays).
Inference on these models is relatively less compute intensive but is more
limited by GPU VRAM, which usually still requires a distributed cluster.
Smaller models (e.g. 7B parameter) and advances in quantization and related
compression techniques have inference (and sometimes training) running on
single machines (including your phone!), sometimes even without GPUs.

Retrieval-Augmented Generation
------------------------------

`Retrieval-Augmented Generation (RAG)
<https://eugeneyan.com/writing/llm-patterns/#retrieval-augmented-generation-to-add-knowledge>`__
enhances a large language model by first retrieving relevant data and adding
it to the input to improve results.  This technique is typically used in a
question and answering scenario.  The name is fancier than it sounds (at least
for the main concept).  LangChain has a good summary on its `Question Answering
Over Documents <https://docs.langchain.com/docs/use-cases/qa-docs>`__ page that
is roughly summarized below.

For the setup, you build an index of your documents where each entry 
is an `embedding <https://en.wikipedia.org/wiki/Word_embedding>`__  
that represents a chunk of text (e.g. several paragraphs).  In
more detail:

1. Due to the limitations of LLMs, you will typically split your documents into
   bite-sized chunks that fit into the LLMs context window (e.g. 4K tokens).
2. Using the LLM, create an embedding from each of your chunks.
3. Store the embedding in a vector store that can retrieve similar
   vectors based on a given input vector (e.g. find the top-K matching
   chunks for a given embedded input query).

Once you have a vector store populated, answering proceeds as follows:

1. Take the input question and convert it to an embedding.
2. Look up top-K relevant entries in your vector store.
3. Construct a prompt based on the input question and these chunks.
4. Send the prompt to an LLM and return the result.

The original `RAG paper <https://arxiv.org/abs/2005.11401>`__ was written
before LLMs got really powerful so it seems that they do a bunch of other
fancy tricks.  However with LLMs, you don't need to do much more than
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
modification of the weights can be for a subset of the layers, all layers,
or even none of them but with some additional trainable augmentations to the
model.  Variants of the latter have been a `popular technique
<https://arxiv.org/abs/2106.09685>`__ to cheaply fine-tune an existing LLM,
reducing the cost by orders of magnitude compared to training the base model
(or naively directly fine-tuning an LLM).  Typically the fine-tuning uses a
lower learning rate so you retain a substantial portion of the learning of the
pre-trained model.

The previously mentioned "instruction fine-tuning" or "human alignment" steps
are a form of fine-tuning where the base language model is only good at
predicting the next token, but fine-tuning it gives you the ability to follow
instructions and respond as humans would expect (vs. just predicting the next
most likely token).  Another example of fine-tuning is training with more
specific data for a task (e.g. Medical Q&A), which has shown to improve
performance over generic models.

OpenAI and LangChain APIs
-------------------------

Most of you will be familiar with `OpenAI <https://openai.com/>`__, most likely
from their breakout product `ChatGPT <https://chat.openai.com/>`__ that was probably
the first widespread demonstration of what LLMs could do (particularly because it
could follow instructions).  What's probably also obvious to most people is that
OpenAI has many `APIs <https://platform.openai.com/docs/introduction>`__ that
allow programmatic access to all of the functionalities of ChatGPT and more.

The APIs are HTTP endpoints that have officially released libraries for
Python and Node.js (as well as other community maintained ones).  The most relevant
APIs are the `chat` and `completions` endpoints which to respond
to a prompt, and the fine-tuning API to fine-tune a model on your own data.  The cost
is usually priced per 1000 tokens for both chat/completion APIs and fine-tuning.
The latter charges different rates for training and inference depending on the
model.

For most of their language APIs, you can select which model you want to use.  The models
are roughly binned into how powerful each one is with the original ChatGPT
release named as `gpt-3.5-turbo`.  The current most powerful model is named
`gpt-4` and they also have many others from older generations of GPT-3 models.

Working with the OpenAI APIs is pretty straightforward, but often times you want
additional functionality (such as RAG) and `LangChain <https://www.langchain.com/>`__
is one of the *many* libraries that fills in the gap.  It appears to be one of the
first and thus relatively popular at the moment, but things are changing fast.
LangChain has a Python library and a more recent JavaScript one, both of which
I used in this project.

The main advantage of LangChain (in my opinion) is that they have many predefined
patterns that you can put together such as RAG.  They have numerous examples
along with the building blocks you need to set up a default LLM application
with components such as predefined prompts, inclusion of various vector
databases, and integration with all popular LLM provider libraries.  It's hard to
say if this will be *the* LLM library of the future but it's definitely a useful
library to get up and running quickly.

Cloudflare Workers
------------------
`Workers <https://workers.cloudflare.com/>`__ is a serverless code platform
developed by Cloudflare.  Although the large cloud providers (also known as
hyperscalers) generally have a serverless code offering (e.g. AWS Lambda), Cloudflare
touts several advantages such as:

* Automatic scaling 
* High performance
* Low latency startup time
* Better developer experience (DX)

One of the fundamental ideas is that you shouldn't have to think about the underlying
infrastructure at all, just deploy and have it work (e.g., no selecting region
or instance size).

Of course, these benefits do come with trade-offs.  Their serverless code 
`runs in V8 isolates <https://developers.cloudflare.com/workers/learning/how-workers-works/>`__,
the same technology that Chrome's JavaScript engine uses to sandbox
each browser tab, and enables Workers to have high performance and low
latency.  The obvious limitation here is that it primarily is focused on JavaScript.
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
external database via HTTP, but the platform didn't natively support
persisting data.  Cloudflare has been pushing hard on the innovation to make their solution
full stack by including things such as a zero-egress fee S3-compatible object store `R2 <https://www.cloudflare.com/developer-platform/r2/>`__, 
an eventually consistent key value store `Workers KV <https://www.cloudflare.com/developer-platform/workers-kv/>`__, 
a serverless SQL database `D1 <https://developers.cloudflare.com/d1/>`__, and
a transaction store with `Durable Objects <https://developers.cloudflare.com/durable-objects/>`__.
Some of these are still in beta but Cloudflare's track record is pretty good at
building thoughtful additions to their platform with good DX.  It remains to be
seen if they can truly disrupt the established hyperscaler dominance.


ROUGE Metric
------------

The `ROUGE <https://en.wikipedia.org/wiki/ROUGE_(metric)>`__ or Recall-Oriented
Understudy for Gisting Evaluation is a family of metrics to evaluate
summarization and machine translation NLP tasks.  They work by comparing
the automatically generated proposed (i.e., *hypothesis*) text to one or more *reference* texts
(usually human generated).  In general, evaluation of NLP tasks is hard because
it heavily depends on the meaning of the text, which historicaly was very hard
to discern (at least before the LLM revolution).  Instead of tackling this head on,
researchers developed simpler mechanical metrics such as ROUGE that
do not depend on the meaning.

ROUGE has many different variants with the simplest one called `ROUGE-N` being
based on the overlap of `N-grams <https://en.wikipedia.org/wiki/N-gram>`__
(word level) between the hypothesis text (:math:`s_{hyp}`) and reference text
(:math:`s_{ref}`) given by the formula:

.. math::

   \text{ROUGE-N} = \frac{\big| \text{N-GRAM}(s_{hyp}) \cap \text{N-GRAM}(s_{ref}) \big|}{\big|\text{N-GRAM}(s_{ref})\big|} \tag{2}

where :math:`\text{N-GRAM}(\cdot)` generates the multiset of (word-level) n-gram tokens and the
intersection operates on multisets, and the :math:`|\cdot|` indicated cardinality of the multiset.

Since we're using :math:`s_{ref}` in the denominator, it's a recall oriented
metric.  However, we could just as well use :math:`s_{hyp}` in the denominator
and it would be the symmetrical precision oriented metric.  Similarly, 
we could compute the related `F1-score <https://en.wikipedia.org/wiki/F-score>`__
with these two values.  This is one of the evaluation metrics that I'll use
later on to give a rough idea of how good the LLM performed.

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

LLM Evaluation using LLMs
-------------------------

As we saw above with the ROUGE metric, evaluation of models up until recently
mainly focused on mechanical metrics.  With the advent of powerful models though,
we can do better by using a *stronger* LLM to evaluate our target LLM performance.
A common method is to use GPT-4 (the current state of the art) to evaluate
whatever LLM task you are working on.  In general because it's so strong
at understanding the semantic meaning of text, it can perform quite well
compared to a human (at least as far as we can tell) and sometimes even better.
The only problem is that the state of the art (GPT-4) can't really be evaluated
using itself for obvious reasons.  That's not so much of a problem in this post
because I only used earlier generation models (mostly due to cost but also
earlier on due to the lack of availability of GPT-4).

Project Details
===============

This section gives an overview of the project components and highlights some of the details
that are not apparent from the code.  
All the `code is available <https://github.com/bjlkeng/bjlkengbot>`__ on Github
but please keep in mind that it's a one-off so I know it's a mess and don't
expect anyone really to use it again (including myself).  I also deployed the
code so anyone could ask LLM-me a question (we'll see how long it takes before
the OpenAI APIs I use get deprecated): `bjlkengbot.bjlkeng.io <https://bjlkengbot.bjlkeng.io/>`__.

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
sentences, and then words, roughly keeping semantically related pieces
together.  You can additionally utilize the OpenAI tokenizer using :code:`from_tiktoken_encoder()`
to match the token counts that OpenAI's API expects.
A chunk size of 900 tokens with 100 overlapping tokens was used.  These numbers
were chosen because I planned to send 4 documents into the RAG workflow so
I wanted it to be less than the default 4096 token window for the ChatGPT-3.5
endpoint.

All of this was done as a pre-processing step because (as we will see later) the
LangChain JavaScript library doesn't (at the time of writing to my knowledge)
have the specific splitter + OpenAI tokenizer.  So splitting
the text into the appropriate chunks first allowed me to not have to worry about
doing much manipulation in JavaScript.  The resulting output was a JSON file
containing an array of objects with the chunked text, and the associated
URL/title metadata for each chunk.

Embeddings and Vector DB
------------------------

With the data collected and chunked, the next step is to implement the RAG pattern.
Luckily LangChain and LangChain.js have some builtin flows to help with that.
The usual flow is to index all your documents which involves: 

1. Creating :code:`Document` objects
2. Connecting to an embedding model (e.g. :code:`OpenAIEmbeddings`)
3. Retrieving embeddings for each document and indexing them in a vector store
4. Persisting the vector store (if not using an online database)

Then for inference, you simply:

1. Load the vector store (if needed)
2. Embed input question using LLM and search for relevant docs in vector store
3. Create a prompt using the input question and retrieved docs
4. Ask LLM the prompt and return response

Since I wanted to deploy the model inference to Cloudflare, I had to use 
LangChain.js for both indexing and inference.  This would have been fine except
that Cloudflare has some quirks.  
Although Cloudflare Workers `mostly supports <https://developers.cloudflare.com/workers/runtime-apis/nodejs/>`__ 
a `Node <https://nodejs.org/en>`__ environment there is (at least) one major
difference: there is `no filesystem <https://developers.cloudflare.com/workers/learning/security-model/>`__.  
This is part of their security model to prevent security issues.  Fair enough.
But this posed a slight challenge because all of LangChain.js memory vector
model stores only support serializing to disk (I didn't want to use a full blown DB).
After thinking for a bit, I realized that almost all objects in JavaScript can
be serialized trivially with :code:`JSON.stringify()`, so I just accessed the
internal vector store storage and serialized that to a file.  That file would
then be stored on R2 (object store), which then could be read back in a Worker
(not using LangChain.js) and I could construct a new vector store object and
just assign the internal storage.  This worked out pretty well (and much better
than my initial naive idea of reindexing the whole corpus on every inference
call).

In terms of the LangChain.js API, it was pretty simple to index using
:code:`MemoryVectorStore.fromDocuments()`, and inference was also a breeze using 
the :code:`RetrievalQAChain`.  I must say that the documentation for these wasn't great
so I often had to look at the implementation to figure out what was going on.
Thank goodness for open source.

In terms of models, I used OpenAI's :code:`text-embedding-ada-002` for embeddings,
and :code:`gpt-3.5-turbo` (ChatGPT-3.5 endpoint) for completion.  With the aforementioned,
4 chunks x 900 token per chunk plus a max token generation of 256, I didn't
have too much trouble fitting into the 4096 token limit of the model.  The
only other parameter I changed from default was a temperature of 0.2.  I 
didn't really try any other values because I just wanted something sufficiently
low to not get totally different answers each time.

My prompt was relatively simple where I took some parts from the default
:code:`RetrievalQAChain` prompt:

.. code::

    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that "I am not programmed to answer that", don't try to make up an answer.
    
    ### Context ###
    Your name is Brian Keng.
    
    {context}
    
    ### Question ###
    Question: {question}
    Helpful answer in less than 100 words:

I supposed I could have improved the prompt with extra background information
about myself but I was lazy and didn't think it was worth it.

    
Fine Tuning
-----------

The other method I played with was using the OpenAI API for fine-tuning.
This *sort of* fits in the `example <https://platform.openai.com/docs/guides/fine-tuning/fine-tuning-examples>`__ 
use-cases on the OpenAI website where they recommends fine tuning for setting a "style and
tone" (the other major use-case is for structured output).
The biggest issue with what I want to do is that my corpus is still just a set
of blog posts, which actually matches the RAG pattern the best.  But I did want
to see if fine-tuning could help capture more of my writing style and tone.

At the time of implementation, the fine-tuning API was not instruction tuned
so it would *only* try to do a completion without the "smarts" about
understanding an instruction.  Due to the expensive cost (at the time),
I used the :code:`curie` model instead of the more expensive :code:`davinci` one.

.. admonition:: LLM Development Is Fast Moving

    To show how fast things have been changing, they don't offer fine-tuning
    with :code:`curie` models any more, and they added instruction tuned
    :code:`gpt-3.5` (ChatGPT) with GPT4 coming along soon.  
    Further, due to instruction tuned versions being the recommended fine-tuning
    model, some of the pre-processing isn't even applicable anymore.  
    For anything to do with LLMs in the next year or two, you probably
    want to look up the source documentation instead of any second hand account
    (like this post) lest it be out of date.

The biggest problem with trying out fine-tuning was that I didn't have
a good dataset!  All I had was a bunch of text, but I wanted to build a
Q&A bot so I needed questions and answers.  Luckily, LLMs are very adaptable,
so I used the ChatGPT API to generate questions from the snippets of my blog!

To do this I first chunked my blog posts (and excluded some of the non-relevant chunks) to
250 tokens using the above mentioned OpenAI :code:`Tiktoken` encoder.  This
mostly chunks it into paragraphs since I typically write short paragraphs.

Next, I prompted the ChatGPT (GPT 3.5) API with the following:

.. code::

    Write a concise question in as few words as possible to the author in the second person that has the following TEXT as the answer.

    ### TEXT ###

where the text chunk is appended to the prompt.  The prompt is pretty self
explanatory, except for the :code:`###` demarcations.  This is a trick
to help the LLM separate the instruction from the "data".  I didn't play
around with it much but it seems like it's a pretty standard prompting trick.

The fine-tuning format (for the older version of OpenAI fine-tuning that I
used) required a clear separator to end the *prompt* and the *completion*
required a white space to start with a clear ending token.  For the former
I used :code:`\n\n###\n\n`, and the latter I used :code:`END`.  Additionally,
each training sample should be put in a JSONL format.  Here's an example line:

.. code::

   {
      "prompt": "QUESTION: Is 2022 feeling more like a \"normal\" year for you?\n\n###\n\n",
      "completion": " Thankfully 2022 has felt a bit more like a “normal” year.  ... END"
   }

This little dataset generation script ran pretty smoothly with the only added
tweak was to add rate limiting since OpenAI doesn't like you hammering their
API.

Once I had the dataset ready in the required format, it was pretty straightforward
to use OpenAI's CLI to fine tune.  The main hyperparameters I played with were
`epochs`, `learning_rate_multiplier`, and `batch_size`.  
When you call the API, it queues up a fine-tuning job and you can poll an API
to see it's status.  My jobs typically trained overnight.  The job also has
an associated ID that you can use when you want to call it for inference.
The only thing to remember is that you need to add the above separators to
ensure that your questions have the same format as during training.


Cloudflare Worker
-----------------

The Cloudflare Worker was pretty straightforward to put together.
The parts that I spent the most time on were (a) learning modern Javascript,
and (b) figuring out how to call the relevant libraries.
The Worker is simply a async Javascript function that Cloudflare
uses to respond to a request.  With their :code:`wrangler` workflow,
it was pretty easy to get it deployed.

The RAG flow was the more complicated one where in addition to calling
OpenAI, I had to load the serialized :code:`MemoryVectorStore` from 
R2 (which took some time to figure out but otherwise has simple code). 
The rest of the flow was easily done using LangChain.js using the appropriate APIs.
The fine-tune flow simply consisted of calling the OpenAI API with
my selected model.

The one thing I will call out is that to test/debug the endpoint, I deployed
it each time.  There is a local server you can spin up to emulate the code
but I didn't really take the time to figure out how to get that working for R2.
I suspect if you're using a lot of the Cloudflare ecosystem (especially the
newer services), it will be increasingly difficult to do local development.
On the other hand, it only took an additional 20 seconds to deploy but having
not needed to "compile" anything since my C++ programming days, it felt like a pain. 

Web Page 
--------

The web page is basic HTML with client side Javascript to call the Cloudflare
Worker endpoint.  It's hosted on Cloudflare pages, which is basically a similar
service to Github pages except with a lot of extra integration into Cloudflare
services.  It was pretty easy to setup, and it has a full continuous deployment
flow where a commit triggers the page to be updated.

Truthfully, getting the page to do what I wanted was a pain in the arse and
took a long time!  I have some rudimentary knowledge of CSS but it just also
feels so fiddly and I had a lot of trouble getting things just right (even with
my super simple ugly page).  On top of that, it's hard to Google for the exact problem
you have since I would only find basic examples that didn't address my specific problem.
However, ChatGPT came to the rescue!  It didn't generate it in one
go, but I asked it to write a basic example of what I wanted, which then served
as a template for me to modify and create the final page.

A couple of other random experiences.  It's no wonder that modern pages use
some kind of Javascript framework.  Even with the handful of UI elements I had
on the page, I had to start maintaining state so that they would all work
together.  I definitely appreciate modern pages a lot more now, but I will say that
the work is not suited to me.  Maybe it's because I've only worked on more
algorithmic type systems but web development seems so foreign to me.

The other point I'll mention is that this type of web development benefits a
lot from local development.  At first I was iterating by just pushing to
Github, which is relatively fast (< 1 mins to update).  But when I'm trying to
get the positioning right of a UI element by playing with the style sheets,
it's not the right flow.  I played around with the browser inspector to debug /
prototype, but inevitably you have to deploy to see if it works.  I finally bit
the bullet and figured out how to set it up locally, which was trivial because
it's just a static HTML page!  I ended up just accessing the local copy from
my web browser.


Model Evaluation
================

To evaluate the model, I used the training dataset from the fine tuned section,
which includes questions that were generated using ChatGPT-3.5 from snippets of
the original blog posts.  This pseudo-Q&A dataset is not at all ideal for evaluation
because I'm using the exact same dataset to fine-tuning the
models.  The other reason it's not ideal is that these questions and answers
are not completely in agreement because the question is LLM generated and the
answer is a chunk of my blog post, not an actual answer.  Despite this, it
was the easiest way to generate an evaluation dataset and I believe gives a
flavour of the results you can expect (but not at all scientific).  In total,
there were 669 Q&As in the dataset.

The models I compared were the standard RAG flow plus differently fine-tuned
OpenAI :code:`curie` (non-instruct) models.  :code:`curie` is a smaller model compared to the
(then largest) :code:`davinci` GPT-3 model on OpenAI.  This was primarily used because of cost.
I originally tried to fine-tune :code:`davinci` and (at the time) I calculated it would
have blew through my `$50` budget.  I ended up spending a bit under `$100` after all
the iterations, which would have been much more if I had used the larger model.

For each model, I generated the answer from the selected question using the
prompts above, then compared the results versus the reference answer on two
categories of metrics. 

ROUGE Metrics
-------------
The first set of metrics use ROUGE with the ROUGE-1,
ROUGE-2 and ROUGE-L F1 variants.  The results are shown in Table 1.

.. csv-table:: Table 1: Mean ROUGE evaluated performance for RAG and Fine-Tuning Models
   :header: Model,"Num Epochs","Batch size","LR Multiplier","ROUGE-1 F1","ROUGE-2 F1","ROUGE-L F1"
   :align: center

   RAG,N/A,N/A,N/A ,0.3311,0.1455,0.3055
   Fine-tune (Curie),2,1,0.05,0.2279,0.0540,0.2093
   Fine-tune (Curie),2,1,0.10,0.2356,0.0598,0.2170
   Fine-tune (Curie),2,1,0.20,0.2552,0.0690,0.2350
   Fine-tune (Curie),2,5,0.10,0.2244,0.0510,0.2049
   Fine-tune (Curie),4,1,0.05,0.2548,0.0679,0.2348
   Fine-tune (Curie),4,1,0.10,0.2714,0.0794,0.2494
   Fine-tune (Curie),4,1,0.20,**0.3382**,**0.1494**,**0.3157**
   Fine-tune (Curie),4,5,0.10,0.2434,0.0565,0.2226

As you can see, the fine-tuned Curie model with 4 epochs, batch size 1 and
learning rate multiplier of 0.20 performed the best with ROUGE metrics of
0.3382, 0.1494, and 0.3157.  The RAG solution is not too far behind with
0.3311, 0.1455, and 0.3055 respectively.  Interestingly, the other fine-tuned
models performed significantly worse, which shows that the hyperparameters
for fine-tuning matter a lot.


LLM Evaluation
--------------

As we know ROUGE is a very crude metric that only depends on n-grams in the
text and doesn't evaluate the semantic meaning.  So next I tried the LLM route
to evaluate the answers using both GPT-3.5 (:code:`text-davinci-003`) and GPT-4.  
Given the above answers, I prompted GPT-3.5 using the following prompt
using with the `Guidance <https://github.com/guidance-ai/guidance>`__ library:

.. code::

   QUESTION: {{question}}

   ANSWER: {{reference}}

   PROPOSED ANSWER: {{hypothesis}}

   Can you rate the PROPOSED ANSWER to the above QUESTION from 0 (not even close) to 10 (exact meaning) on whether or not it matches ANSWER?  Only output the number.
   {{select 'rating' options=valid_nums logprobs='logprobs'}}

The nice thing about guidance is that you can easily insert templates but most uniquely, you can guide the
generation.  So for example the :code:`{{select ... options=valid_nums}}`
constrains the output to the valid numbers (in this case between 0 and 10).  It also allows you to extract
the log probabilities, which I generated and then calculated the expected value
(mean) of the resulting distribution.  Note: It probably doesn't make sense
to use GPT-3.5 to evaluate a GPT-3.5 output in the case of RAG, but perhaps
makes sense for the smaller :code:`curie` model?

Similarly, I did a similar exercise for GPT-4 using the following prompt:

.. code::

   {{#system~}}
   You are a helpful assistant.
   {{~/system}}
   {{#user~}}
   QUESTION: {{question}}
   
   ANSWER: {{reference}}
   
   PROPOSED ANSWER: {{hypothesis}}
   
   Can you rate the PROPOSED ANSWER to the above QUESTION from 0 (not even close) to 10 (exact meaning) on whether or not it matches ANSWER?  Only output the number.
   {{~/user}}
   {{#assistant~}}
   {{gen 'rating' temperature=0 max_tokens=2}}
   {{~/assistant}}

Note that GPT-4 is a conversational endpoint so it has the added system/user/assistant functionality.
Additionally, these endpoints don't provide log probabilities (either as input or output) so you can't
use the Guidance library constraints with them.  The final value output here is
simply the numeric tokens from 0 to 10 where I limited the tokens to 2 so
it wouldn't give me additional spurious output.  The results of these two
experiments are in Table 2.

.. csv-table:: Table 2: Mean GPT-3.5/4 evaluated performance on a 0 to 10 scale for RAG and Best Fine-Tune Models
   :header: Model,"GPT-3.5","GPT-4","Wins","Ties"
   :widths: 12,7,6,5,4
   :align: center

   RAG,**8.746**,**7.643**,**486**,102
   Fine-tune (best),8.048,4.872,81,102

As you can see the LLM outputs paint a different picture compared to the ROUGE scores.  Here RAG
shows a much better match to the reference answer than the fine-tuned model.  This can be seen
in the GPT-4 rating of 7.643 vs. 4.872 respectively, and within this difference the
RAG models has a higher rating (i.e., "wins") 486 times to fine-tuned 81 times
with 102 ties.  The GPT-3.5 output seems a bit less trustworthy because of the
reasoning above and shows the gap being much smaller.

Example Output
--------------

Taking a look at some of the examples, we can see some anomalies in the data based on the non-ideal
experimental setup we have.  First, let's take a look a good RAG example and a poor fine-tuned example:

.. code::

   QUESTION: What are some examples of contexts where memorization is beneficial?

   REFERENCE: 
   As with many things in life, I feel like I’ve come full circle.  Recently, I’ve started taking Chinese classes again as well as music lessons (mostly guitar, a bit of vocal).  One thing that is crystal clear is that memorization is
   hugely beneficial… in certain contexts.  For example, my Chinese speaking has always been lacking.  I was a shy kid and didn’t really make much of an effort to speak in Chinese.  However by memorizing and reciting the textbook lessons, character by character, I’ve been told my Chinese has improved quite a bit.  There’s something about actually moving your mouth, activating your vocal chords, and having sound come out that gets your neurons connecting properly.  Another more obvious explanation: if I wanted to get better at speaking, I should practice speaking more!  However, I will point out that the opportunities to memorize a piece of text and say it out loud are much more abundant than speaking to an actual human.  In any case, a very useful application of memorization.

   RAG (Score: 10):
   Memorization is beneficial in contexts where physical practice is needed, such as speaking a language or playing a musical instrument. It is also beneficial in contexts where it allows you to ignore the "easy" parts and focus on the important parts, such as memorizing the lyrics to a song so you can focus on the actual singing, or memorizing the multiplication table so you can focus on algebra.

   FINE-TUNED (Score: 0):   Memorization is definitely one of those things that you either have a talent for or you don’t.  I definitely have a talent for it, and I’ve been using it for most of my academic career (since high school actually).  Here are a few examples of contexts where it’s beneficial:

You can see here that the RAG nicely summarized the result while the fine-tuned
model just went in a totally different direction. This is characteristic of the
fine-tuned model in certain cases where I suppose it didn't "remember" the
fine-tuning and is basing its answer on its pre-training.  Interestingly, the
RAG result gives examples that weren't in the reference answer (although they
may have been in some of my other posts)! 

Going in the opposite direction, we can see where the fine-tuning did well and
the RAG did not:

.. code::

   QUESTION: Who wrote "My Life"?
   
   REFERENCE:  My Life (Bill Clinton)

   RAG (Score: 0):
   The book "Models of My Life" was written by Herbert A. Simon. It is an autobiography looking at his distinguished multi-disciplinary career filled with tidbits of incredibly valuable wisdom throughout. It is a candid and witty autobiography that I highly recommend reading.

   FINE-TUNED (Score: 10):   My Life (Bill Clinton)

Here you can see that the fine-tuned model just memorized the training data (a
limitation of my non-ideal setup).  The RAG answer on the other hand conflated
the Bill Clinton book "My Life" and "Models of My Life" written by Herbert A.
Simon.  Both books I have written about and it appears that the LLM could not
appropriately distinguish the "my life" in these two contexts.

A couple of more funny examples where both models scored well according to GPT-4:

.. code::

   QUESTION: Did John continue to increase the voltage despite the student's cries of pain?

   REFERENCE:  
   John felt a bit apprehensive about this whole situation but Dr. Milgram reassured him that this was safe. So he began with the first word on the list. The student from the next room answered incorrectly. John pressed the red button and the student made a sound that was barely audible. He turned the dial up 50V and proceeded to the next word on the list. The student again responded incorrectly so he pressed pressed the red button again. This time the student could clearly be heard from the next room. John pressed on. Turn up 50V and ask another word. This time the student got it right, John was relieved but that soon faded as the student from the next room cried out in pain from his incorrect answer.

   RAG (Score: 10):
   Yes, John continued to increase the voltage despite the student's cries of pain. Dr. Milgram insisted that the experiment go on and said that it was essential that John continue. John felt apprehensive about the situation, but he followed Dr. Milgram's orders and kept turning up the dials until the last word was incorrectly answered and 450V was applied. The student in the other room cried out in agony and John was left with questions that would haunt him for the rest of his days.

   FINE-TUNED (Score: 10):
   John kept increasing the voltage.  The student kept crying out in pain but John didn’t stop.  He just kept going until the student finally stopped yelling.  He then told John to increase the voltage again.  This time, he yelled even louder.  But John didn’t stop.  He just kept going until he yelled “no more”.  Then he told John to increase the voltage again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.  And again.

Here the fine-tuned model went a bit off the rails repeating the same two words
over and over again, which kind of makes sense in the context but is probably
not correct.  Interestingly, GPT-4 thought it was still a pretty good answer
probably because the meaning is still about correct.  Another quirk that the
fine-tuning model picked up on was that I put two spaces after a period.  In
the RAG flow (since I didn't ask it to), it uses the more popular one space
after period.  So I guess fine-tuning did learn something about style!

.. code::

   QUESTION: Does sleep have a measurable effect on happiness?

   REFERENCE:  
   The Chinese proverb at the beginning of the post is not too far off.  Although it may not bring a lifetime of happiness, sleep does have a measurable effect on happiness.  The way I look at it, it’s more of a necessary but not sufficient condition for happiness (sorry for the logic speak).  In other words, if you’re tried, you will be significantly less happy.  But at the same time having enough sleep doesn’t guarantee happiness either.  I think this is true in many other respects too such as being hungry or being healthy.  If you’re hungry, most likely you’re not happy.  If you’re sick, most likely you’re not happy.  The biggest difference with sleep, however, is that many people neglect to value sleep as highly as the other two (obesity seems to be a growing problem but sleeping in isn’t).  But with sleep, I at least have the capacity to be happy (not to mention think clearly and more productively).  However, as with most problems (weight loss comes to mind), the problem isn’t understanding the idea, it’s implementing it.

   RAG (Score: 10):
   Yes, sleep has a measurable effect on happiness. Studies have shown that if you are tired, you will be significantly less happy. Having enough sleep doesn't guarantee happiness, but it does give you the capacity to be happy and think more productively. If you want to increase your happiness, start with a good night's sleep.

   FINE-TUNE (Score: 10):
   “Sleep has a measurable effect on happiness.”

Here the fine-tuned model is pretty good being super concise but perhaps not
being as helpful as the RAG one.  You can take a look at more of
the examples in this (very messy) `notebook <https://github.com/bjlkeng/bjlkengbot/blob/main/finetune/measure.ipynb>`__.


Commentary
==========

LLMs as Coding Assistants
-------------------------

Through this project (and the one before that I stopped halfway to work on this
one) I've been using ChatGPT (free version), Github Copilot, and to a lesser
extent GPT-4 API via Simon Willison's great `llm <https://github.com/simonw/llm>`__ tool.
And all I can say for sure is that LLMs have a decent noticeable productivity boost.  

For me, the biggest boost was with ChatGPT writing Javascript and HTML.
Ages ago I did a bit of Javascript in "Web 1.0", and then after my PhD I did an
online interactive Javascript book (I can't seem to find it but it was pretty
good) but that also was over a decade ago, suffice it to say that I hadn't done
any modern web development for a while.  

In learning modern Javascript, ChatGPT was incredibly helpful.  I had a strong
idea of what I wanted to accomplish, knew most of the primitives in the
language, but I was unclear on some of the details.  For example, I asked ChatGPT
to explain :code:`let` vs. :code:`var` vs. no declaration (had a bug related to
it).  Module imports were another new thing (as I understand).  And one thing
I found super frustrating was getting the styling (CSS) correct on the HTML (even
though it's super basic).  Getting the spinner to be centered where I wanted it
was incredibly tough without ChatGPT because every search on the web would only
show the most basic example without solving the one annoying issue I had.
It turned out that ChatGPT's "knowledge" and it's chat interface to *specify*
and *respond* more precisely to what I wanted was indeed quite a bit superior
to just a Google search.  It's almost an improved `StackOverflow <https://stackoverflow.com/>`__ in
real time.

Another area where I found it quite useful was producing pretty well known
snippets of code.
In the other project I was working on, I wanted to write a transformer from
scratch and so I asked ChatGPT to generate some PyTorch modules.  As far as I
could tell (I didn't finish the project yet), it looked correct!  Transformer
modules are probably so widespread (even before ChatGPT's 2021 cutoff date) that it
could easily write one.  It did save me some time doing it myself though,
it was similar to having an intern (a common LLM analogy) where I just needed to 
check its work.

On the other hand, I still reverted back to the original docs for the libraries
I worked with.
Things like :code:`langchain` and Cloudflare workers are newer and aren't
encoded in the LLMs knowledge base well (or all).  So really the combination
of manual docs + LLMs is still the best and I believe needed to deliver a
working application.

On the Copilot side, I found it only slightly useful.  It helped do some simple
autocomplete based on the context of my code but it really only helped me
reduce some typing.  It's good for ergonomics, especially with more boiler
plate code, but I wasn't as impressed as compared to ChatGPT.  Still, I
would probably still pay the $10/month for it since it is a small but noticeable
quality of life improvement.

On the GPT-4 front, I was only really using it to do simple tasks
like write birthday cards (and as an evaluation metric above).  I haven't
really used it to its full potential yet because I only have the API 
version now, which doesn't have the data analysis and plugin capability. It's my
default LLM right now when I want to answer a quick question at the command-line and
don't need a chat interface.  I'll probably write more about it when I find
something interesting in my workflow to use it for.

LangChain
---------
`langchain <https://github.com/langchain-ai/langchain>`__ was one of the earliest
LLM frameworks.  It was useful to get things up and running because it takes
care of all of the details from calling the LLM APIs to vector databases to
even simple prompts.  My impression is that it's still immature, as is the
entire area.  It's obvious to me that the API is still clunky and probably not
exactly the abstraction you want to build these types of applications.

The other thing that annoyed me is that the documentation wasn't detailed
enough.  Maybe it's just my habit of wanting to understand a lot of detail
when I call an API but I found myself having to look at the source code
and reading through it to be able to use it properly.  Thank goodness for
open source!  The days where I had to reverse engineer how to get certain
Windows APIs to work are long gone (to be fair MSDN had very good documentation 
for its day).

VSCode vs. Vim/Jupyter
----------------------
The other change I made for this project was to switch over to VSCode.  I've
been a Vim user for over fifteen years so I was very reluctant.
Of course VSCode has Vim emulation but there is always something that is a bit
off whenever it was implemented.  My reasoning for switching was that the
integration with Github Copilot and Jupyter notebooks would be worth it.

My overall impression, sad as it might be, is that it probably makes sense
for me to switch over to use it.  Besides the boost you get from Copilot,
having notebooks in the same IDE, and the superior code navigation, it also has
great support for remote development, which was always an advantage Vim had
over other IDEs.

I'm still not completely used to VSCode though, particularly the vertical splitting
of the screen, which I did a lot in Vim.  And the notebook keyboard shortcuts
still throw me off as there are certain actions I still haven't figured out how
to use keyboard for. Nonetheless, I'm sure I'll adapt to it in time (longer now
that I'm not coding everyday though).

Cloudflare
----------
As I mentioned above, I haven't really done much web development at all.  So I
just had cursory knowledge of a lot of the services that Cloudflare provides.
I have to say it was super easy to get setup considering my limited knowledge.

Workers was easy enough to get working having an in-browser IDE to play around
with.  It took a bit more setup to get a local version working (in VSCode) that 
could deploy with a command but not that much more work with the documentation
and tutorials.  The ability to easily connect to R2 object store was also quite
nice, which just involved adding the name to the config file and then using
the attached environment variable in the JS program. 

Beyond moving some of my domains over to Cloudflare, I also used the (free)
DDoS protection to rate limit the number of connection to the above site
because it calls my OpenAI account which costs money.  It was pretty easy
to set up with a few clicks and it seems to work reasonably well.

All of the above (besides the domain registration) would have been free if it
was not for the fact that the worker call needs some non-zero CPU time to
execute.  As such, I signed up for the $5/month plan, which like the free plan,
is so generous that I basically won't need to pay more.

LLMs: Do we need to worry?
---------------------------

So after playing around with LLMs for a bit, what's the conclusion?  In
general, I think there's more hype than is justified in the first year or so.
LLMs aren't going to mass replace jobs (yet), and they are definitely far away
from general intelligence.  

But... they are definitely useful.  It's clear that as an interface, it will
improve the way we interact with many computing devices.  The chat interface
is powerful, and as the cost comes down, it will only become more pervasive.
One of the really powerful things is the accessibility it gives to non-coders.
I can just imagine (in some later better UX) my mom using something that is
powered by an LLM in the background to do some "programming".  Think of a Star
Trek kind of computer interface.  Of course there will be many challenges like
hallucinations, safety, and privacy, but it's not a big leap to see how things
will change.

What's not clear to me though is if there is something bigger than that around
the corner.
The obvious tasks like summarization, Q&A, and conversational agents all 
have started to permeate through many applications.  The real question here is
if there is another killer app that we haven't yet encountered (or perhaps
haven't yet discovered the necessary technology unlock for).  
Human ingenuity is boundless so I suspect there will be something in a few
years where we will be saying "I can't believe we didn't think of that."
In the meantime I'm pretty confident that my job isn't going to go away and
will only get easier (assuming they allow us to use LLMs at work). 


Conclusion
==========

So that's my little project on LLMs.  It was a good learning experience hitting a
few things that I wanted to learn more about with one stone.  There are many obvious
places where I could improve the project like using the latest versions of
OpenAI models, using a combination of fine-tuning and RAG patterns, or generating a better
dataset, but honestly, I'm not that interested in doing more.  I'm generally a
late adopter to many things because I don't want to "waste" my time on fads.
That might just be my age showing (although I'm not that old).  My personality
biases towards going deep on time tested ideas.  I guess I'm just not built to
keep up with the latest trends.
It was a fun project though and I might end up doing more of these "building"
projects instead of diving deep into the math and algorithms.  That's the
beauty of this site, I can do whatever I want!  See you next time.


Further Reading
===============

* `A High-Level Overview of Large Language Models <https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/>`__
* `Building LLM-based Systems & Products <https://eugeneyan.com/writing/llm-patterns/#retrieval-augmented-generation-to-add-knowledge>`__


.. [1] In fact, there are several projects going on at work that are related to this topic but since I'm in a technical management role, I spend almost no time coding or directly doing research.  Thus, this blog is my outlet to satisfy both my curiosity and to help stay current.
