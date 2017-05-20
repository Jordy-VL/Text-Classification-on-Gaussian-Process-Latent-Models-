import nltk
import csv
import sklearn
import numpy as np
import pandas as pd
from sklearn.naive_bayes import *
from pandas import Series,DataFrame
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from nltk.tag.perceptron import PerceptronTagger
from sklearn.preprocessing import *
import re
from nltk.stem import *
from nltk.classify import *
from textblob import TextBlob
import json
from nltk.tokenize import *
from nltk.stem import *
from nltk.classify import *
from sklearn.feature_extraction.text import *
from nltk.corpus import wordnet
from nltk.classify import ClassifierI
from statistics import mode
from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.util import *
import re
import math
import random
from random import shuffle

tweets = []
sentiment = []

trainEndInd = 5000
testEndInd = 8000

#opening training data set file 
dataS = []

# with open('subj_data_0.5.json', 'r') as jsonfile:
#     data = json.load(jsonfile)
#     for i in data:
#         tweets.append(i['tweet'])
#         sentiment.append(i['sentiment'])
#         dataS.append((i['tweet'].strip(' \n\"'), i['sentiment'].strip(' \n\"')))
                     
                     
    
# with open('data.csv', 'r', encoding='utf-8', errors='ignore') as csvfile:
#     tweetData = csv.reader(csvfile, delimiter=',', quotechar='|')
#     #retreiving the tweet and the sentiment 
#     for row in tweetData:
#         sentiment.append(row[0].strip(' \n\"'))
#         tweets.append(row[5].strip(' \n\"'))
#         dataS.append((row[0].strip(' \n\"'), row[5].strip(' \n

filetext = open('datatext.txt', 'r')
filelabel = open('datalabel.txt', 'r')

for lines in filetext.readlines():
    tweets.append(lines)

for labels in filelabel.readlines():
    sentiment.append(int(labels))

random.shuffle(dataS)

# tweets = []
# sentiment = []
# datasize = 20000
# for x in dataS:
#     if len(tweets) > datasize:
#         break
#     tweets.append(x[0])
#     sentiment.append(x[1])
tweets = np.array(tweets)
sentiment = np.array(sentiment)

X_train, X_test, y_train, y_test = train_test_split(tweets, sentiment, random_state=0, test_size=0)

stemmer = PorterStemmer()
stemmer1 = SnowballStemmer("english")

precessedData = []
stopWords = []

#contains stop words to be removed
fp = open('StopWords.txt', 'r')
line = fp.readline()
while line:
    word = line.strip()
    stopWords.append(word)
    line = fp.readline()
fp.close()

#storing the acronym dictionary
acroDict1 = []
acroDict2 = []
fp2= open('AcronymDict.txt', 'r')
line = fp2.readline()
while line:
    synonyms = line.split('-')
    acroDict1.append(synonyms[0])
    acroDict2.append(synonyms[1])
    line = fp2.readline()
fp2.close()

#starting the function 
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

def preprocessTweet(Tweet):
    Tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))',' ',Tweet) #strip off URLs
    Tweet = re.sub('@[^\s]+','',Tweet) #removing user tags
    Tweet = re.sub(r'#([^\s]+)', r'\1', Tweet) #replacing hash tag followed by word with just the word
    Tweet = replaceTwoOrMore(Tweet) #look for 2 or more repetitions of character and replace with the character itself 
    Tweet = Tweet.strip(' \n,;:-?!*()\"') #removing punctuations
    Tweet = re.sub('[&*!-?$#^.,:;%<>}{[]/\"]',' ',Tweet) #removing symbols
    Tweet = re.sub('[\s]+','', Tweet) #extra whitespaces handled
    Tweet = Tweet.lower() #convert Tweet to lower case
    a = ':)' 
    b = ':('
    #stripping of emoticons
    Tweet = Tweet.replace(a,'')
    Tweet = Tweet.replace(b,'')
    #performing stemming
    #Tweet = stemmer.stem(Tweet)
    #Tweet = stemmer1.stem(Tweet)
    if Tweet in stopWords:
        return ''
    if len(Tweet) > 0:
        return Tweet
    return ''

def preprocess(tweet):
    """
    This function removes stop words and punctuation with the NLTK tokenize package required for Part B
    """
    tweet = tweet.lower()
    tweet = tweet.strip('{,\" \n')
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(tweet)
    filtered_words = [w.lower() for w in tokens]
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return " ".join(filtered_words)

processedData = []
traintweet = []
for Tweet1 in X_train:
    Tweet1 = re.sub('\.\.+', ' ', Tweet1)
    Tweet1 = re.sub('\-\-+', ' ', Tweet1)
    Tweet1 = re.sub('\.', '', Tweet1) 
    Tweet1 = preprocess(Tweet1)
    str = ""
    for Tweet in Tweet1.split():
        Tweet = preprocessTweet(Tweet)
        if len(Tweet) > 0:
            str = str + Tweet + ' '
    processedData.append(str)
    traintweet.append(str)

#Dont't Run
print(len(y_train))
print(len(processedData))

tweets = np.array(processedData)
sentiment = np.array(y_train)

#tweets, X_test, sentiment, y_test = train_test_split(tweets, sentiment, random_state=2, test_size=0)

count_vect = CountVectorizer(ngram_range = (1, 1), max_features=1000)
X_new_counts = count_vect.fit_transform(tweets)
#tfidf_transformer = TfidfTransformer(use_idf=True)
#X_train_tfidf = tfidf_transformer.fit_transform(X_new_counts)
#X_train_tfidf = X_new_counts
clf = NuSVC().fit(X_new_counts[:800], sentiment[:800])
#clf = RandomForestClassifier(100).fit(X_new_counts[:40000], sentiment[:40000])
#clf = MultinomialNB().fit(X_new_counts[:1000], sentiment[:1000])
#clf = LogisticRegression().fit(X_new_counts[:9000], sentiment[:9000])
predicted= clf.predict(X_new_counts[800:1000])
np.mean(predicted == sentiment[800:1000])

import GPflow
from __future__ import print_function
import GPflow
from GPflow import ekernels
from GPflow import kernels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('ggplot')

#import pods
#pods.datasets.overide_manual_authorize = True  # dont ask to authorize
np.random.seed(42)
GPflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations

x_train, x_test, y_train, y_test = train_test_split(X_new_counts[:1000], sentiment[:1000], random_state=2, test_size=0)

print(x_train.shape)
print(x_test.shape)

model3=RandomForestClassifier(100).fit(x_train[:800],y_train[:800])
accuracy_score(model3.predict(x_train[800:1000]),y_train[800:1000])

Y = x_train[:1000].toarray()
Y = Y.astype(float)
Q = 100
M = 200  # number of inducing pts
N = Y.shape[0]
X_mean = GPflow.gplvm.PCA_reduce(Y, Q) # Initialise via PCA

Z = np.random.permutation(X_mean.copy())[:M]

#k = ekernels.Linear(20, ARD=False)
k = ekernels.RBF(100, ARD=True)
# fHmmm = False
# if(fHmmm):
#     k = ekernels.Add([ekernels.RBF(3, ARD=True, active_dims=slice(0,3)),
#                   ekernels.Linear(2, ARD=False, active_dims=slice(0,20))])
# else:
#     k = ekernels.Add([ekernels.RBF(3, ARD=True, active_dims=[0,1,2]),
#                   ekernels.Linear(2, ARD=False, active_dims=[3, 4])])

m = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean, X_var=0.1*np.ones((N, Q)), Y=Y,
                                kern=k, M=M, Z=Z)
m.likelihood.variance = 0.01
m.optimize(disp=True, maxiter=100)

from sklearn import svm
from sklearn.metrics import accuracy_score

model = svm.LinearSVC()
model.fit(np.array(m.X_mean.value[:800]),y_train[:800])
accuracy_score(model.predict(m.X_mean.value[800:1000]),y_train[800:1000])

model2 = svm.NuSVC()
model2.fit(np.array(m.X_mean.value[:800]),y_train[:800])
accuracy_score(model2.predict(m.X_mean.value[800:1000]),y_train[800:1000])

model3=RandomForestClassifier(100).fit(np.array(m.X_mean.value[:800]),y_train[:800])
accuracy_score(model3.predict(m.X_mean.value[800:1000]),y_train[800:1000])

