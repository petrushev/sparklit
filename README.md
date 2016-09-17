# sparklit
Natural Language Processing with Python and Spark

## api

  * [sentenceSplit](#sentencesplit)
  * [tokenizeSentences](#tokenizesentences)
  * [ginSentences](#ginsentences)
  * [search](#search)
  * [vocabulary](#vocabulary)
  * [tf](#tf), [idf](#idf), [tfidf](#tfidf)


### sentenceSplit
    Takes raw text rdd and transforms it into rdd containing sentences only


### tokenizeSentences
    Takes sentences only rdd and transforms it into rdd of key-value pairs,
    the key being the sentence, and the value being list of tokens


### ginSentences
    Takes sentences only rdd and transforms it into gin index rdd,
    the key is a word and the value is an iterable of sentences


### search
    Takes gin rdd and returns rdd of sentences that contain the token


### vocabulary
    Takes sentence-token list rdd and transforms it into the vocabulary rdd
    containing only the alphabetical tokens from the text


### tf
    Build tf breakdown of sentence-token list. 
    The resulting key is sentence-token pair.


### idf
    Build idf breakdown of sentence-token list.
    The resulting key is sentence.


### tfidf
    Build tf-idf breakdown of sentence-token list.
    The resulting key is sentence-token.


## Running tests

You will need **[nose](nose.readthedocs.io)** to run tests:

   ``nosetests2 test.py``
