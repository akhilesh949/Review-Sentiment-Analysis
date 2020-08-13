# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 01:20:07 2018

@author: Akhilesh

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#quoting = 3 ignores double quotes in our data

#Cleaning text
import re
import nltk
nltk.download('stopwords') #package containing lists of irrelevant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ',  dataset.values[i,0]) #the character which is not a-z or A-Z will be removed and replaced by a space
    review = review.lower()
    review = review.split() #splitting the text into individual words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))] #each word is getting stemmed appended in the array if the word is not in stopwords
    review = ' '.join(review)
    corpus.append(review)

#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer #creating a sparse matrix with each word as a column (TOKENIZATION)
cv = CountVectorizer(max_features = 1500) #includes 1500 most relevant words in the sparse matrix
X = cv.fit_transform(corpus).toarray() #to array to convert into matrix
y = dataset.iloc[:,1].values #dependent variable

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
