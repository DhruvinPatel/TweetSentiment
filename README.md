# TweetSentiment
###Task:
The Assignment consisted of creating a Sentiment Classifier that will predict a given tweet to
have a positive or negative sentiment.

###Requirements:
Neccessary python libraries such as Skicit, NLTK pre-installed.

###How to run:
Since I was not able to upload MODEL.pkl due to its large size (as was initially trained on large data of about 1.6million tweets), you will have to generate one! You fill find training data here: http://www.cse.iitd.ac.in/~mausam/courses/csl772/autumn2014/A2/A2.pdf

Run train.py with two arguments: 1. Number of tweets 2. Input file containing labelled tweets (in format as above specified link).
This will generate MODEL.pkl file.

inputfilename contains list of tweets you wish to classify on.
outputfilename is the file where class label: "0" for negative and "4" for positive corresponding to each tweet will be added. 

Commands: <br>
$ ./compile.sh <br>
$ ./run.sh inputfilename outputfilename


System Description:
A sentiment categorization system using several Machine Learning Techniques for Tweets Data-set
in Python. The MODEL.pkl (in the final system) corresponds to logistic regression as a binary classifier.
Refer to the Writeup for more details.
