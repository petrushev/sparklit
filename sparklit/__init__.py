import string
from operator import itemgetter, add
from math import log
from contextlib import contextmanager

from nltk.data import load as nltkDataload
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.regexp import WordPunctTokenizer

from pyspark.storagelevel import StorageLevel


englishPunktTokenizer = nltkDataload('tokenizers/punkt/english.pickle')
wordPunktTokenizer = WordPunctTokenizer()

def sentenceSplit(textRdd, sentTokenizer=englishPunktTokenizer):
    """
    Takes raw text rdd and transforms it into rdd containing sentences only
    @param sentTokenizer: object with method `tokenize`
    """
    return textRdd.flatMap(englishPunktTokenizer.tokenize)\
                  .map(string.strip).filter(lambda sent: sent != '')

def tokenizeSentences(sentenceRdd, wordTokenizer=wordPunktTokenizer):
    """
    Takes sentences only rdd and transforms it into rdd of key-value pairs,
    the key being the sentence, and the value being list of tokens
    @param wordTokenizer: object with method `tokenize`
    """
    return sentenceRdd.map(lambda sent: (sent, wordTokenizer.tokenize(sent)))

def _filterTokens(tokens):
    """
    Filter out non-alphabetical tokens and put the rest of them in lowercase
    """
    return [token.lower() for token in tokens
            if token.isalpha()]

def ginSentences(sentenceRdd, wordTokenizer=wordPunktTokenizer):
    """
    Takes sentences only rdd and transforms it into gin index rdd,
    the key is a word and the value is an iterable of sentences
    @param wordTokenizer: object with method `tokenize`
    """
    return tokenizeSentences(sentenceRdd, wordTokenizer)\
        .flatMap(lambda (sent, tokens): \
                 [(token, sent) for token in set(_filterTokens(tokens))])\
        .groupByKey()

def search(gin, token):
    """
    Takes gin rdd and returns rdd of sentences that contain the token
    """
    return gin.filter(lambda (_token, sents): _token == token)\
              .flatMap(lambda (_token, sents): sents)

def vocabulary(sentWordMapRdd):
    """
    Takes sentence-token list rdd and transforms it into the vocabulary rdd
    containing only the alphabetical tokens from the text
    """
    return sentWordMapRdd.flatMap(lambda (sent, tokens): _filterTokens(tokens))\
               .map(lambda token: (token, None))\
               .groupByKey().keys()

def _tf(tokens):
    freq = {}
    tokens_ = _filterTokens(tokens)
    for token in tokens_:
        freq[token] = freq.get(token, 0) + 1
    total = len(tokens_)
    tf = [(token, cnt * 1.0 / total)
          for token, cnt in freq.iteritems()]
    return tf

def tf(sentWordMapRdd):
    """Build tf breakdown of sentence-token list.
    The resulting key is sentence-token pair.
    """
    return sentWordMapRdd.flatMap(
        lambda (sent, tokens): [((sent, token), tf_) for token, tf_ in _tf(tokens)])

@contextmanager
def temporaryPersisted(rdd, storageLevel=StorageLevel.DISK_ONLY):
    isPersisted = rdd.getStorageLevel()
    if isPersisted is None:
        unpersist = True
        rdd.persist(StorageLevel.DISK_ONLY)
    else:
        unpersist = False

    yield rdd

    if unpersist:
        rdd.unpersist()

def idf(sentWordMapRdd):
    """Build idf breakdown of sentence-token list.
    The resulting key is sentence
    """
    with temporaryPersisted(sentWordMapRdd) as sentWordMapRdd:
        vocabularySize = vocabulary(sentWordMapRdd).count()
        df = sentWordMapRdd.flatMap(lambda (sent, tokens): \
                                        [(token, 1) for token in set(_filterTokens(tokens))])\
                           .foldByKey(0, add)
        res = df.mapValues(lambda val: log(vocabularySize * 1.0 / val))

    return res

def tfidf(sentWordMapRdd):
    """Build tf-idf breakdown of sentence-token list.
    The resulting key is sentence-token
    """
    with temporaryPersisted(sentWordMapRdd) as sentWordMapRdd:
        tfSent = tf(sentWordMapRdd).map(lambda ((sent, token), val): (token, (sent, val)))
        res = tfSent.join(idf(sentWordMapRdd))\
                     .map(lambda (token, ((sent, tf_), idf_)): ((sent, token), tf_ * idf_))

    return res
