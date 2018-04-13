import nltk 
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression ##model we use
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer() ## turns words into their base form. dogs -> dog.

stopwords = set(w.rstrip() for w in open('stopwords.txt')) ##remove words without a specific meaning

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

##
print 'positive size: ' 
print len(positive_reviews)
print 'negative size: ' 
print len(negative_reviews)
## 983 positive reviews and 402 negative reviews
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]


def my_tokenizer(s):
    #first : lower case the entire string
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    ##turns words in their base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    ##remove stopwords
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1 

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1 

## data array for each token, we will use word porportion again
## create our input matrices
def tokens_to_vector(tokens,label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()
    x[-1] = label ## set the last element to the lable
    return x

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map)+1))
i=0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1) # 1 is the lable for positive reviews
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0) # 0 is the lable for negative reviews
    data[i,:] = xy
    i += 1

##shuffle before getting our train sets
np.random.shuffle(data)

X = data[:, :-1] ##all rows everything except the last colum
Y = data[:,-1] ##lables of the last colum

Xtrain = X[:-100,]
Ytrain = Y[:-100,]

Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
# from sklearn.ensemble import AdaBoostClassifier
# model = AdaBoostClassifier()
model.fit(Xtrain,Ytrain)
print "Classification rate:",model.score(Xtest,Ytest)


# let's look at the weights for each word
# try it with different threshold values!

# threshold = 0.5
# for word, index in iteritems(word_index_map):
#     weight = model.coef_[0][index]
#     if weight > threshold or weight < -threshold:
#         print(word, weight)
    