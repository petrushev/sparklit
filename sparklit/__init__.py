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
    return textRdd.flatMap(englishPunktTokenizer.tokenize)\
                  .map(string.strip).filter(lambda sent: sent != '')

def tokenizeSentences(sentenceRdd, wordTokenizer=wordPunktTokenizer):
    return sentenceRdd.map(lambda sent: (sent, wordTokenizer.tokenize(sent)))

def _filterTokens(tokens):
    return [token.lower() for token in tokens
            if token.isalpha()]

def ginSentences(sentenceRdd, wordTokenizer=wordPunktTokenizer):
    return tokenizeSentences(sentenceRdd, wordTokenizer)\
               .flatMap(lambda (sent, tokens): \
                            [(token, sent) for token in set(_filterTokens(tokens))])\
               .groupByKey()

def search(gin, token):
    return gin.filter(lambda (_token, sents): _token == token)\
              .flatMap(lambda (_token, sents): sents)

def vocabulary(sentWordMapRdd):
    return sentWordMapRdd.flatMap(lambda (sent, tokens): _filterTokens(tokens))\
               .map(lambda token: (token, None))\
               .groupByKey().keys()

def tf(sentWordMapRdd):
    return sentWordMapRdd.flatMap(lambda (sent, tokens): \
                                      [((sent, token), 1) for token in _filterTokens(tokens)])\
                         .foldByKey(0, add)

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
    with temporaryPersisted(sentWordMapRdd) as sentWordMapRdd:
        vocabularySize = vocabulary(sentWordMapRdd).count()
        df = sentWordMapRdd.flatMap(lambda (sent, tokens): \
                                        [(token, 1) for token in set(_filterTokens(tokens))])\
                           .foldByKey(0, add)
        res = df.mapValues(lambda val: log(vocabularySize * 1.0 / val))

    return res

def tfidf(sentWordMapRdd):
    with temporaryPersisted(sentWordMapRdd) as sentWordMapRdd:
        tfSent = tf(sentWordMapRdd).map(lambda ((sent, token), val): (token, (sent, val)))
        res = tfSent.join(idf(sentWordMapRdd))\
                     .map(lambda (token, ((sent, tf_), idf_)): ((sent, token), tf_ * idf_))

    return res
