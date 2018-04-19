import nltk
import random
import numpy as np

from bs4 import BeautifulSoup

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

trigrams = {}

for review in positive_reviews:
    s = review.text.lower() #so we dont get 2 versions of the same word
    tokens = nltk.tokenize.word_tokenize(s)
    for i in xrange(len(tokens)-2): # from 0 till the 3rd last word
        k = (tokens[i],tokens[i+2]) # key is a touple. touples are immutable so they can be keys. lists cannot
        if k not in trigrams:
            trigrams[k] = [] 
        trigrams[k].append(tokens[i+1])

# trigrams:
# (u'recording', u'of'): [u'lots'],
# (u'less', u','): [u'durable', u')', u'camera'],
#
#transform into a probablity vector

trigrams_probabilities = {}
for k, words in trigrams.iteritems():
    if len(set(words)) > 1:
        d={} ##dic , key by the middle word
        n=0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] +=1
            n += 1
        for w,c in d.iteritems():
            d[w] = float(c) / n
        trigrams_probabilities[k] = d

#create a function to randomly sample from these trigrams probablities

def random_sample(d): #dictionary where key is a word, and a value is the prob. of that word
    r = random.random()
    cumulative = 0
    for w, p in d.iteritems():
        cumulative +=p
        if r<cumulative:
            return w

def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print "Original: ", s
    tokens = nltk.tokenize.word_tokenize(s)
    for i in xrange(len(tokens)-2):
        if random.random() < 0.2:
            k = (tokens[i],tokens[i+2])
            if k in trigrams_probabilities:
                w = random_sample(trigrams_probabilities[k])
                tokens[i+1] = w
    print "Spun:" #nltk tokenize punctuation as well.
    print " ".join(tokens).replace(" .", ".").replace(" '", "'").replace("$ ","$").replace(" !", "!")

#examples:
# Original:
# excellent antenna.  why pay so much more for other indoor antennas such as terk or samsung. i tried both of those but ended up with the zenith at a fraction of the price and with much better reception

# Spun:
# excellent antenna. why pay so much use for other indoor antennas such as terk or mappable. i tried both of those but ended up to the zenith at a fraction of the price and with much better reception

# In [21]: test_spinner()
# Original:
# this is a great swicth with low price.  i use it together with d-link di-704 4-port internet sharing router to build my home network with 1 pc, two servers, one powermac and a laptop connected to cable modem isp. connecting via the swicth for internet sharing doesn't show any noticeable slow-response as compared to connecting via the router directly.  it works well by itself as we as with d-link di-704

# Spun:
# this is a great swicth with low price. i use it together with d-link di-704 4-port internet sharing router to get my home network with 1 pc , two servers , one powermac and a laptop connected to cable modem isp. connecting via the swicth for internet sharing does n't even any noticeable slow-response as compared to connecting via the router directly. it works well by humans as we as with d-link di-704

# conclusion:
# most of the time does not make sense:
# there should be more context checking  than just the 2 neighboring words.
# our markov-like assumption does not hold.
# this is just 5% of what we need to do in order to build a good article spinner

