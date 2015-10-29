import sys
import os
import pdb
import csv
import numpy
import scipy
import sklearn
#print sklearn.__file__ 
import random
from random import shuffle
from datetime import datetime
import pickle
import re
#coding: utf-8. -*-


tstart = datetime.now()
csv.field_size_limit(sys.maxsize)

INPUT_FILE_NAME = sys.argv[1]
OUTPUT_FILE_NAME = sys.argv[2]
FILE_NAME = 'MODEL.pkl'#'Bernoullinb_withnot_3.pkl'

in_file = open(INPUT_FILE_NAME,"r")
lines = in_file.readlines()
NUM_TEST = len(lines)

pos = {}
neg = {}
#commonwords=['the','a','i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs']
commonwords = ['tis','\'twas','a','able','about','across','after','all','almost','also','am','among','an','and','any','are','aren\'t','as','at','be','because','been','but','by','can','could','could\'ve','dear','did','didn\'t','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','he\'d','he\'ll','he\'s','her','hers','him','his','how','how\'d','how\'ll','how\'s','however','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','it','it\'s','its','just','least','let','like','likely','may','me','might','might\'ve','mightn\'t','most','must','must\'ve','mustn\'t','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','she\'d','she\'ll','she\'s','should','should\'ve','since','so','some','than','that','that\'ll','that\'s','the','their','them','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','tis','to','too','twas','us','wants','was','wasn\'t','we','we\'d','we\'ll','we\'re','were','weren\'t','what','what\'d','what\'s','when','when','when\'d','when\'ll','when\'s','where','where\'d','where\'ll','where\'s','which','while','who','who\'d','who\'ll','who\'s','whom','why','why\'d','why\'ll','why\'s','will','with','won\'t','would','would\'ve','wouldn\'t','yet','you','you\'d','you\'ll','you\'re','you\'ve','your']

def ascii_conv (l):
	ans = ''
	for i in l:
		if(ord(i)<=128):
			ans = ans + i #append(i)
	return ans
#print lines[:2]

test_lines = []

for l in lines:
	l = unicode(l, errors='ignore')
	test_lines.append( l.strip('\n') )

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer() 

def return_modified (line):
	ln = line.split(' ')
	lns =ln[0]
	for l in range(len(ln)-1):
		#  check re with ln[i]
		if re.match( r'^(not)|(aren\'t)|(can\'t)|(couldn\'t)|(didn\'t)|(doesn\'t)|(don\'t)|(hadn\'t)|(hasn\'t)|(haven\'t)|(isn\'t)|(mightn\'t)|(mustn\'t)|(needn\'t)|(not\'ve)|(shan\'t)|(shouldn\'t)|(shouldn\'t\'ve)|(wasn\'t)|(weren\'t)|(won\'t)|(wouldn\'t)|(wouldn\'t\'ve)|(hadn\'t\'ve)|(ain\'t)$', ln[l+1] ):
			if(l+2<len(ln)):
				ln[l+2] = 'qwerty'+ln[l+2]
		
		else:	
			lns = lns + ' ' + lmtzr.lemmatize(ln[l+1])
	return lns		
test = []
for t in test_lines:
	test.append(return_modified (t))

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
 }

from sklearn.externals import joblib

def save_classifier(classifier):
   f = open('A2_my_classifier.pickle', 'wb')
   joblib.dump(classifier, f, compress = -9)#
   f.close()

def load_classifier():
   f = open(FILE_NAME, 'rb')
   classifier = pickle.load(f)
   f.close()
   return classifier


import numpy as np
text_clf = load_classifier()
predicted = text_clf.predict(test)#(X_new_count)

f = open(OUTPUT_FILE_NAME,'w')
for i in range(NUM_TEST):
	f.write(predicted[i]+'\n')
f.close()


	


print datetime.now() - tstart

