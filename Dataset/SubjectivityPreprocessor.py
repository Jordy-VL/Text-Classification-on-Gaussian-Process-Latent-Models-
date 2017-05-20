import csv
from textblob import TextBlob
import json

jsondata = []
subjectivity_threshould = 0.5

#opening training data set file
with open('data.csv', encoding='utf-8', errors='ignore') as csvfile:
  tweetData = csv.reader(csvfile, delimiter=',', quotechar='|')
  #retreiving the tweet and the sentiment
  for row in tweetData:
    try:
      tweetblob = TextBlob(row[5])
      if tweetblob.sentiment.subjectivity > subjectivity_threshould:
        tweet = {}
        tweet['sentiment'] = row[0]
        tweet['tweet'] = row[5]
        jsondata.append(tweet)
    except Exception as e:
      print (e)

with open('subj_data_0.5.json', 'w') as jsonfile:
  json.dump(jsondata, jsonfile, indent=4)

