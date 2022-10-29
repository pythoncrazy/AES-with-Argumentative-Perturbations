Natural Language Toolkit
========================

NLTK is a leading platform for building Python programs to work with human language data.
It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet,
along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

Thanks to a hands-on guide introducing programming fundamentals alongside topics in computational linguistics,
NLTK is suitable for linguists, engineers, students, educators, researchers, and industry users alike.
NLTK is available for Windows, Mac OS X, and Linux. Best of all, NLTK is a free, open source, community-driven project.

NLTK has been called "a wonderful tool for teaching, and working in, computational linguistics using Python,"
and "an amazing library to play with natural language."

`Natural Language Processing with Python <https://sites.google.com/site/naturallanguagetoolkit/book>`_ provides a practical
introduction to programming for language processing.
Written by the creators of NLTK, it guides the reader through the fundamentals
of writing Python programs, working with corpora, categorizing text, analyzing linguistic structure,
and more.

Some simple things you can do with NLTK
---------------------------------------

Tokenize and tag some text:

    >>> import nltk
    >>> sentence = """At eight o'clock on Thursday morning
    ... Arthur didn't feel very good."""
    >>> tokens = nltk.word_tokenize(sentence)
    >>> tokens
    ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning',
    'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
    >>> tagged = nltk.pos_tag(tokens)
    >>> tagged[0:6]
    [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('on', 'IN'),
    ('Thursday', 'NNP'), ('morning', 'NN')]

Identify named entities:

    >>> entities = nltk.chunk.ne_chunk(tagged)
    >>> entities
    Tree('S', [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'),
               ('on', 'IN'), ('Thursday', 'NNP'), ('morning', 'NN'),
           Tree('PERSON', [('Arthur', 'NNP')]),
               ('did', 'VBD'), ("n't", 'RB'), ('feel', 'VB'),
               ('very', 'RB'), ('good', 'JJ'), ('.', '.')])

Display a parse tree:

.. doctest::
    :options: +SKIP

    >>> from nltk.corpus import treebank
    >>> t = treebank.parsed_sents('wsj_0001.mrg')[0]
    >>> t.draw()

.. image:: images/tree.gif

Links
-----

* NLTK-Users mailing list: http://groups.google.com/group/nltk-users
* NLTK's previous website: https://sites.google.com/site/naturallanguagetoolkit
* NLTK development: https://github.com/nltk
* NLTK-Dev mailing list: http://groups.google.com/group/nltk-dev
* Publications about NLTK: http://scholar.google.com.au/scholar?q=NLTK

Contents
========

.. toctree::
   :maxdepth: 1

   news
   install
   data
   api/nltk

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
