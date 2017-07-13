.. title: Building A Table Tennis Ranking Model
.. slug: building-a-table-tennis-ranking-model
.. date: 2017-07-18 08:51:41 UTC-04:00
.. tags: Bradley-Terry, ranking, ping pong, table tennis, Rubikloud
.. category: 
.. link: 
.. description: A post on the Bradley-Terry Model for pair-wise ranking.
.. type: text

I wrote a couple of posts about building a table tennis ranking model
over at Rubikloud:

* `Building A Table Tennis Ranking Model <https://rubikloud.com/labs/building-table-tennis-ranking-model/>`__.  

It uses
`Bradley-Terry <https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model>`__
probability model to predict the outcome of pair-wise comparisons (e.g. games
or matches).  I talk about an easy algorithm for fitting the model (via the
MM-algorithms) as well as adding a simple Bayesian prior to handle ill-defined
cases.  I even have some 
`code on Github <https://github.com/bjlkeng/Bradley-Terry-Model>`__ 
so you can build your own ranking system using Google sheets.

Here's a blurb:

    Many of our Rubikrew are big fans of table tennis, in fact, we’ve held an
    annual table tennis tournament for all the employees for three years
    running (and I’m the reigning champion). It’s an incredibly fun event where
    everyone in the company gets involved from the tournament participants to
    the spectators who provide lively play-by-play commentary.
    
    Unfortunately, not everyone gets to participate either due to travel and
    scheduling issues, or by the fact that they miss the actual tournament
    period in the case of our interns and co-op students. Another downside is
    that the event is a single-elimination tournament, so while it has a clear
    winner the ranking of the participants is not clear.
    
    Being a data scientist, I identified this as a thorny issue for our
    Rubikrew table tennis players. So, I did what any data scientist would do
    and I built a model.

Enjoy!
