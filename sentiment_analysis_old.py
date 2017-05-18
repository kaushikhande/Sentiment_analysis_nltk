import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import csv
import numpy as np
posFeatures = []
negFeatures = []

def load_file():
    with open('review.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        posdata = []
        negdata = []
        for row in reader:
            # skip missing data
            if row[0] and row[1]:
                data_review_tuple = row[0]
                
                #target.append(row[1])
                if row[1] == 'positive':
                    posdata.append(data_review_tuple)
                else:
                    negdata.append(data_review_tuple)

        return posdata,negdata

posdata,negdata = load_file()

def word_split(data):    
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new

#this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select):
    negfeats = [(feature_select(f), 'neg') for f in word_split(negdata)]
    #print len(negfeats)
    posfeats = [(feature_select(f), 'pos') for f in word_split(posdata)]    
	#selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posfeats)*3/4))
    #print posCutoff 
    negCutoff = int(math.floor(len(negfeats)*3/4))
    trainFeatures = posfeats[:posCutoff] + negfeats[:negCutoff]
    #print len(trainFeatures)
    testFeatures = posfeats[posCutoff:] + negfeats[negCutoff:]
    
	#trains a Naive Bayes Classifier	
    classifier = NaiveBayesClassifier.train(trainFeatures)	

	#initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)	

	#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
		referenceSets[label].add(i)
		predicted = classifier.classify(features)
		testSets[predicted].add(i)	

	#prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', nltk.precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', nltk.recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', nltk.precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', nltk.recall(referenceSets['neg'], testSets['neg'])
    print classifier.show_most_informative_features(10)

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
	return dict([(word, True) for word in words])

#tries using all words as the feature selection mechanism
print 'using all words as features'
evaluate_features(make_full_dict)
#scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
	#creates lists of all positive and negative words
	posWords = []
	negWords = []
	posRev,negRev = load_file()
	#print posWords
	#raw_input('> ')
	#posWords = list(itertools.chain(*posWords))
	#negWords = list(itertools.chain(*negWords))
	#negWords = [(make_full_dict(f), 'neg') for f in word_split(negWords)]
        #print len(negfeats)
        #posWords = [(make_full_dict(f), 'pos') for f in word_split(posWords)]
        
        for f in word_split(negRev):
            posWords.append(f)
        
        for f in word_split(posRev):
            negWords.append(f)
    #build frequency distibution of all words and then frequency distributions of words within positive and negative labels
        word_fd = FreqDist()
	cond_word_fd = ConditionalFreqDist()
	#print posWords
	posWords = list(itertools.chain(*posWords))
	negWords = list(itertools.chain(*negWords))
	for word in posWords:
	    word_fd[word.lower()] += 1
	    #print word
	    #raw_input('>')
	    cond_word_fd['pos'][word.lower()] += 1
        #count = count + 1
	for word in negWords:
		word_fd[word.lower()] += 1
		cond_word_fd['neg'][word.lower()] += 1

	#finds the number of positive and negative words, as well as the total number of words
	pos_word_count = cond_word_fd['pos'].N()
	neg_word_count = cond_word_fd['neg'].N()
	#print count
	total_word_count = pos_word_count + neg_word_count
        #print total_word_count
        #raw_input('>')

	#builds dictionary of word scores based on chi-squared test
	word_scores = {}
	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
		word_scores[word] = pos_score + neg_score

	return word_scores

#finds word scores
word_scores = create_word_scores()
#print word_scores
#raw_input('>')

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

#numbers of features to select
numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
	print 'evaluating best %d word features' % (num)
	best_words = find_best_words(word_scores, num)
	evaluate_features(best_word_features)

