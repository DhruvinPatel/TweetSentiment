import sys
import os
import pdb
import csv
import numpy
import scipy
import sklearn
import pickle
#print sklearn.__file__ 
import random
from random import shuffle
from datetime import datetime

#coding: utf-8. -*-


tstart = datetime.now()
csv.field_size_limit(sys.maxsize)

#fo = open("workfile.txt", "wb")
#with open('training.csv', 'rb') as csvfile:
#	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#	for row in spamreader:
#		fo.write ('\n'.join(row))

NUM_TWEETS = sys.argv[1]
MODEL_FILE_NAME = "MODEL"

if(len(sys.argv)>2):
	INPUT_FILE = sys.argv[2]
else:
	raise ValueError('Input file not provided')

in_file = open(INPUT_FILE,"r")
lines = in_file.readlines()

pos = {}
neg = {}
# implement re,not_,filter

#commonwords=['the','a','i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs']
commonwords = ['tis','\'twas','a','able','about','across','after','all','almost','also','am','among','an','and','any','are','aren\'t','as','at','be','because','been','but','by','can','could','could\'ve','dear','did','didn\'t','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','he\'d','he\'ll','he\'s','her','hers','him','his','how','how\'d','how\'ll','how\'s','however','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','it','it\'s','its','just','least','let','like','likely','may','me','might','might\'ve','mightn\'t','most','must','must\'ve','mustn\'t','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','she\'d','she\'ll','she\'s','should','should\'ve','since','so','some','than','that','that\'ll','that\'s','the','their','them','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','tis','to','too','twas','us','wants','was','wasn\'t','we','we\'d','we\'ll','we\'re','were','weren\'t','what','what\'d','what\'s','when','when','when\'d','when\'ll','when\'s','where','where\'d','where\'ll','where\'s','which','while','who','who\'d','who\'ll','who\'s','whom','why','why\'d','why\'ll','why\'s','will','with','won\'t','would','would\'ve','wouldn\'t','yet','you','you\'d','you\'ll','you\'re','you\'ve','your']

def ascii_conv (l):
	ans = ''
	for i in l:
		if(ord(i)<=128):
			ans = ans + i #append(i)
	return ans

def nosymbols(char): 
	if not ((ord(char) >= 48 and ord(char)<=57) or (ord(char) >= 97 and ord(char) <= 122)) and char!=' ': 
		return '' 																										
	else: 
		return char
#print lines[:2]

train_lines = []
# = list([(l[5:-1].strip('\n') for l in (  lines [:int(NUM_TWEETS)] + lines [-int(NUM_TWEETS):] ) )])   #+ (l[5:-1].strip('\n') for l in lines [-int(NUM_TWEETS):])
#test_lines = (l[5:-1].strip('\n') for l in lines [-NUM_TWEETS:])

for l in lines[:int(NUM_TWEETS)]:
	#data=l
	#udata=data.decode("utf-8")
	#asciidata=udata.encode("ascii","ignore")
	l = unicode(l, errors='ignore')
	train_lines.append( l[5:-1].strip('\n') )
for l in lines[-int(NUM_TWEETS):]:
	#data=l
	#udata=data.decode("utf-8")
	#asciidata=udata.encode("ascii","ignore")

	#l.encode("ascii", "ignore")
	l = unicode(l, errors='ignore')
	train_lines.append( l[5:-1].strip('\n') )

#print (train_lines)

train_target = ['0']*int(NUM_TWEETS) + ['4']*int(NUM_TWEETS)

test_count_ = int(int(NUM_TWEETS)*0.2*2)
train_count_ = 2*int(NUM_TWEETS) - test_count_

modtw_shuf = []
target_shuf = []
mod_test_shuf = []
target_test_shuf = []

	

index_shuf = range(len(train_lines))#range(len(train_lines)*0.8)
shuffle(index_shuf)
a=0
for i in index_shuf:
	if a<train_count_ :
		modtw_shuf.append(train_lines[i])
		target_shuf.append(train_target[i])
	else:
		mod_test_shuf.append(train_lines[i])
		target_test_shuf.append(train_target[i])
	a=a+1

#print modtw_shuf
print datetime.now() - tstart


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words = 'english' ,min_df = 2, max_df = 0.5 , ngram_range= (1,2))#removing stopwords:																	 stop_words = commonwords, 

X_train_counts = count_vect.fit_transform(modtw_shuf)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

X_new = X_train_counts# SelectKBest(chi2, k=).fit_transform(X_train_counts, target_shuf)
#X_perc = SelectPercentile(chi2,percentile = 10).fit(X_train_counts,target_shuf)
#X_new = X_perc.fit_transform(X_train_counts,target_shuf)
# scikits.learn.feature_selection.univariate_selection.SelectPercentile(score_func, percentile=10)

print 'shape: '
print X_new.shape # for naive bayes = 586781 # with count vector paras : 191048
print datetime.now() - tstart
#print X_new

#print ' '
#print count_vect.vocabulary_.get(u'day')

from sklearn.feature_extraction.text import TfidfTransformer
#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
#print X_train_tf.shape
#(2257, 35788)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = X_new#tfidf_transformer.fit_transform(X_new)#X_train_counts
#print X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC


clf = BernoulliNB().fit(X_train_tfidf.toarray(), target_shuf)

#from sklearn.pipeline import Pipeline
#text_clf = Pipeline([('vect', CountVectorizer(stop_words = commonwords, min_df = 2, max_df = 0.5 )),# try min = 2; max = 0.4; ngram(1,2) commonwords/'english'
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', SVC(kernel = 'linear')),#SVC(kernel = 'linear')
#])
#text_clf = text_clf.fit(modtw_shuf, target_shuf)

#predicted = text_clf.predict(mod_test_shuf)
#for doc, category in zip(mod_test_shuf, predicted):
#	print('%r => %s' % (doc, category))

from sklearn.grid_search import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
 }

#def save_classifier(classifier):
#   f = open('A2_my_classifier.pickle', 'wb')
#   pickle.dump(classifier, f, -1)
#   f.close()

#def load_classifier():
#   f = open('A2_my_classifier.pickle', 'rb')
#   classifier = pickle.load(f)
#   f.close()
#   return classifier

from sklearn.externals import joblib

def save_classifier(classifier):
   f = open(MODEL_FILE_NAME+'.pkl', 'wb')
   #joblib.dump(classifier, 'A2_Naive_classifier.pkl', compress = -9)#
   pickle.dump(classifier, f)
   f.close()

def save_vector(vector):
   f = open(MODEL_FILE_NAME+'_vector'+'.pkl', 'wb')
   #joblib.dump(classifier, 'A2_Naive_classifier.pkl', compress = -9)#
   pickle.dump(vector, f)
   f.close()

def load_classifier():
   f = open('A2_my_classifier.pkl', 'rb')
   classifier = joblib.load(f)
   f.close()
   return classifier


#gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
#gs_clf = gs_clf.fit(modtw_shuf[:int(int(NUM_TWEETS)*0.1)], target_shuf[:int(int(NUM_TWEETS)*0.1)])

#best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
#for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))



#clf__alpha: 0.001
#tfidf__use_idf: True
#vect__ngram_range: (1, 1)

#print score   


#twenty_train.target_names[gs_clf.predict(['God is love'])]

import numpy as np
#twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

#use it to store only svm

#*
if(len(sys.argv)>2):
	save_classifier(clf)
	save_vector(count_vect)
#docs_test = mod_test_shuf
#predicted = text_clf.predict(mod_test_shuf)
#print np.mean(predicted == target_test_shuf)

#from sklearn import metrics
#print(metrics.classification_report(target_test_shuf, predicted,
#    target_names=['0','4']))
#*	


#import pickle
#s = pickle.dumps(clf)

#print text_clf.CountVectorizer().shape
docs_new = mod_test_shuf
X_new_counts = count_vect.transform(docs_new)
X_new = X_new_counts# SelectKBest(chi2, k=).fit_transform(X_train_counts, target_shuf)
X_new_tfidf = X_new#tfidf_transformer.transform(X_new)
#X_new_tfidf = X_perc.transform(X_new)


predicted = clf.predict(X_new_tfidf)
print np.mean(predicted == target_test_shuf)


#for doc, category in zip(docs_new, predicted):
#	print('%r => %s' % (doc, category)) 


#for line in range(len(lines)):
#	if(lines[line][1]=='0'):
#		tline = (lines[line])[5:-1].strip('\n')	#current_tweet=(L[i])[5:-1].strip('\n')
#		words = tline.split()#lines[line].split();
#		for i in range(len(words)):
#			if (neg.has_key(words[i])):
#				neg[words[i]] = neg[words[i]] + 1;	
#			else:
#				neg[words[i]] = 1;
		#fo.write(lines[line]);
#	else:	
		#words = lines[line].split();
#		tline = (lines[line])[5:-1].strip('\n')	#current_tweet=(L[i])[5:-1].strip('\n')
#		words = tline.split()#lines[line].split();
#		for i in range(len(words)):
#			if (pos.has_key(words[i])):
#				pos[words[i]] = pos[words[i]] + 1;	
#			else:
#				pos[words[i]] = 1;
		#fo.write(lines[line])

#fo = open("pos&neg.txt", "wb")		
		
#for i in pos.keys():
#	fo.write(i+':'+str(pos[i])+'\n')
#fo.write('\n\n\n\n\n\n');	
#for i in neg.keys():
#	fo.write(i+':'+str(neg[i])+'\n')	
	
#in_file.close()		
#fo.close()		
print len(modtw_shuf)
print datetime.now() - tstart

