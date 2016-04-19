import sys
import os

def addPysparkPath():
    try:
        os.environ["PYSPARK_PYTHON"] = "/usr/bin/python2"
        spark_home = os.environ['SPARK_HOME']
    except KeyError:
        print "SPARK_HOME not set"
        sys.exit(1)

    # path for pyspark and py4j
    spark_pylib = os.path.join(spark_home, "python", "lib")
    py4jlib = [ziplib
               for ziplib in os.listdir(spark_pylib)
               if ziplib.startswith('py4j') and ziplib.endswith('.zip')][0]
    py4jlib = os.path.join(spark_pylib, py4jlib)
    sys.path.append(os.path.join(spark_home, "python"))
    sys.path.append(py4jlib)

addPysparkPath()

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.storagelevel import StorageLevel

conf = SparkConf()
conf.setMaster('local[*]').setAppName('SparkLit test')
sc = SparkContext(conf=conf)
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

import sparklit

suite = {}


def setUp():
    PATH = 'tests/grace_dubliners_james_joyce.txt'
    data = sc.textFile(PATH, 4)
    data.persist(StorageLevel.MEMORY_ONLY)
    suite['data'] = data

def testLoad():
    data = suite['data']
    assert len(data.collect()) == 309

def testSentSplit():
    data = suite['data']
    sents = sparklit.sentenceSplit(data)
    sents.persist(StorageLevel.MEMORY_ONLY)
    fourthSent = sents.take(4)[3]
    assert fourthSent == u'He lay curled up at the foot of the stairs down which he had fallen.', fourthSent
    assert sents.count() == 723, sents.count()
    suite['sents'] = sents
    suite['fourthSent'] = fourthSent

def testTokenizeSents():
    sents = suite['sents']

    fourthSent = suite['fourthSent']
    tokens = fourthSent.split(' ')
    tokens[-1] = u'fallen'
    tokens.append('.')

    sentWords = sparklit.tokenizeSentences(sents)
    sentWords.persist(StorageLevel.MEMORY_ONLY)
    fourthTokens = sentWords.take(4)[3]

    assert fourthTokens == (fourthSent, tokens), fourthTokens
    suite['sentWords'] = sentWords

def testVocabulary():
    sentWords = suite['sentWords']
    vocab = sparklit.vocabulary(sentWords)
    vocabSet = set(vocab.collect())
    assert set(u'he,lay,curled,fallen'.split(',')).issubset(vocabSet)
    assert vocab.count() == 1767, vocab.count()

def testGin():
    sents = suite['sents']
    fourthSent = suite['fourthSent']

    gin = sparklit.ginSentences(sents)
    gin.persist(StorageLevel.MEMORY_ONLY)
    fallenSents = gin.filter(lambda (token, sents): token == u'fallen')
    fallenSents = fallenSents.collect()

    assert len(fallenSents) == 1, len(fallenSents)
    token, sents = fallenSents[0]
    assert token == u'fallen', token
    sents = set(sents)
    assert fourthSent in sents, sents

    suite['gin'] = gin

def testSearch():
    gin = suite['gin']
    fourthSent = suite['fourthSent']
    q = sparklit.search(gin, u'fallen').collect()

    assert fourthSent in q, q

def testTfIdf():
    sentWords = suite['sentWords']
    fourthSent = suite['fourthSent']
    with sparklit.temporaryPersisted(sparklit.tfidf(sentWords)) as tfidf:
        fourthTfidf = tfidf.filter(lambda ((sent, token), val): sent == fourthSent)\
                           .map(lambda ((sent, token), val): (token, val))\
                           .sortBy(keyfunc=lambda (token, val): -val)
        important, _ = fourthTfidf.first()

    assert important == u'curled', important

def tearDown():
    suite['data'].unpersist()
    suite['sents'].unpersist()
    suite['sentWords'].unpersist()
    suite['gin'].unpersist()
