import collections
import nltk
import csv
import numpy as np
import nltk.classify.util, nltk.metrics
from sklearn import cross_validation
from sklearn.svm import LinearSVC, SVC
import random
from nltk.corpus import stopwords
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures
from nltk.sentiment.util import mark_negation
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
from nltk.probability import FreqDist, ConditionalFreqDist
from sklearn.metrics import confusion_matrix
#posdata = []
#with open('positive-data.csv', 'rb') as myfile:    
#    reader = csv.reader(myfile, delimiter=',')
#    for val in reader:
#        posdata.append(val[0])        
 
#negdata = []
#with open('negative-data.csv', 'rb') as myfile:    
#    reader = csv.reader(myfile, delimiter=',')
#    for val in reader:
#        negdata.append(val[0])            

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
#print len(posdata)
#print len(negdata)

def word_split(data):    
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new
 
def word_split_sentiment(data):
    data_new = []
    for (word, sentiment) in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.a((word_filter, sentiment))
    return data_new
    
def word_feats(words):    
    return dict([(word, True) for word in words])
 
stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'))
     
def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])
 
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    """
    print words
    for ngram in itertools.chain(words, bigrams): 
        if ngram not in stopset: 
            print ngram
    exit()
    """    
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
    
def trigram_word_feats(words, score_fn=TrigramAssocMeasures.chi_sq, n=50):
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigrams = trigram_finder.nbest(score_fn, n)
    """
    print words
    for ngram in itertools.chain(words, bigrams): 
        if ngram not in stopset: 
            print ngram
    exit()
    """    
    return dict([(ngram, True) for ngram in itertools.chain(words, trigrams)])
    

#def bigram_word_feats_stopwords(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
def bigram_word_feats_stopwords(words, score_fn=BigramAssocMeasures.mi_like, n=50):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    #print bigrams
    """
    print words
    for ngram in itertools.chain(words, bigrams): 
        if ngram not in stopset: 
            print ngram
    exit()
    """    
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams) if ngram not in stopset])
 
# Calculating Precision, Recall & F-measure
svm_accuracy = []
maxent_accuracy = []
nb_accuracy = []


def evaluate_classifier(featx):
    
    #negfeats = [(featx(mark_negation(f)), 'neg') for f in word_split(negdata)]
    #posfeats = [(featx(mark_negation(f)), 'pos') for f in word_split(posdata)]
    negfeats = [(featx(f), 'neg') for f in word_split(negdata)]
    #print negfeats[1:25]
    #raw_input('>')
    posfeats = [(featx(f), 'pos') for f in word_split(posdata)]    
    negcutoff = len(negfeats)*3/4
    poscutoff = len(posfeats)*3/4
 
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    print(len(trainfeats))
    #print trainfeats
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print(len(testfeats))
    
    # using 3 classifiers
    classifier_list = ['nb', 'svm', 'maxent']#     
    NB_pred = []
    new_label = []    
    for cl in classifier_list:
        if cl == 'maxent':
            classifierName = 'Maximum Entropy'
            classifier = MaxentClassifier.train(trainfeats, 'GIS', trace=0, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 1)
        elif cl == 'svm':
            classifierName = 'SVM'
            classifier = SklearnClassifier(LinearSVC(), sparse=False)
            classifier.train(trainfeats)
        else:
            classifierName = 'Naive Bayes'
            classifier = NaiveBayesClassifier.train(trainfeats)
            
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        original_label = []
        
 
        for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
                original_label.append(label)
                #print feats
                #raw_input('> ')
                observed = classifier.classify(feats)
                NB_pred.append(observed)
                    
                
                
                
                testsets[observed].add(i)
 
        #print refsets['pos']
        #print testsets['pos']
        #print original_label
        #print NB_Pred
        #cm = confusion_matrix(original_label,NB_pred)
        #print cm
        #print "The accuracy score is {:.2%}".format(accuracy_score(original_label,NB_pred))
        new_label = original_label
        accuracy = nltk.classify.util.accuracy(classifier, testfeats)
        pos_precision = nltk.precision(refsets['pos'], testsets['pos'])
        pos_recall = nltk.recall(refsets['pos'], testsets['pos'])
        pos_fmeasure = nltk.f_measure(refsets['pos'], testsets['pos'])
        neg_precision = nltk.precision(refsets['neg'], testsets['neg'])
        neg_recall = nltk.recall(refsets['neg'], testsets['neg'])
        neg_fmeasure =  nltk.f_measure(refsets['neg'], testsets['neg'])
        
        print('')
        print('---------------------------------------')
        print('SINGLE FOLD RESULT ' + '(' + classifierName + ')')
        print('---------------------------------------')
        print('accuracy:', accuracy)
        print('precision', (pos_precision + neg_precision) / 2)
        print('recall', (pos_recall + neg_recall) / 2)
        print('f-measure', (pos_fmeasure + neg_fmeasure) / 2)    
                
        #classifier.show_most_informative_features(50)
    
    print('')
    
    print len(NB_pred)
    
    ME_pred = NB_pred[982:]
    SVM_pred = NB_pred[491:982]
    NB_pred = NB_pred[0:491]
    #print NB_pred
    #print "-----------------------"
    #print ME_pred
    #print "-----------------------"
    #print SVM_pred
    #print "-----------------------"
    #cm = confusion_matrix(SVM_pred,NB_pred)
    #print cm
    #print "The accuracy score is {:.2%}".format(accuracy_score(SVM_pred,NB_pred))
    #cm = confusion_matrix(ME_pred,NB_pred)
    #print cm
    #print "The accuracy score is {:.2%}".format(accuracy_score(ME_pred,NB_pred))
    #cm = confusion_matrix(SVM_pred,ME_pred)
    #print cm
    #print "The accuracy score is {:.2%}".format(accuracy_score(SVM_pred,ME_pred))
    
    final_pred = []
    for i in range(0,491):
        c1 = 0
        if NB_pred[i] == 'pos':
            c1 = c1 + 1
        if ME_pred[i] == 'pos':
            c1 = c1 + 1
        if SVM_pred[i] == 'pos':
            c1 = c1 + 1
        #print i
        if c1 == 3 or c1 == 2:
            final_pred.append('pos')
        else:
            final_pred.append('neg')
        
    print "-----------------------"
    #print final_pred
    print "-----------------------"
    #print new_label
    cm = confusion_matrix(final_pred,new_label)
    print cm
    print "The accuracy score is {:.2%}".format(accuracy_score(final_pred,new_label))
    

    ## CROSS VALIDATION
    
    trainfeats = negfeats + posfeats    
    
    # SHUFFLE TRAIN SET
    # As in cross validation, the test chunk might have only negative or only positive data    
    random.shuffle(trainfeats)    
    n = 5 # 5-fold cross-validation    
    
    for cl in classifier_list:
        
        subset_size = len(trainfeats) / n
        accuracy = []
        pos_precision = []
        pos_recall = []
        neg_precision = []
        neg_recall = []
        pos_fmeasure = []
        neg_fmeasure = []
        cv_count = 1
        for i in range(n):        
            testing_this_round = trainfeats[i*subset_size:][:subset_size]
            training_this_round = trainfeats[:i*subset_size] + trainfeats[(i+1)*subset_size:]
            
            if cl == 'maxent':
                classifierName = 'Maximum Entropy'
                classifier = MaxentClassifier.train(training_this_round, 'GIS', trace=0, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 1)
            elif cl == 'svm':
                classifierName = 'SVM'
                classifier = SklearnClassifier(LinearSVC(), sparse=False)
                classifier.train(training_this_round)
            else:
                classifierName = 'Naive Bayes'
                classifier = NaiveBayesClassifier.train(training_this_round)
                    
            refsets = collections.defaultdict(set)
            testsets = collections.defaultdict(set)
            for i, (feats, label) in enumerate(testing_this_round):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)
            
            cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)
            cv_pos_precision = nltk.precision(refsets['pos'], testsets['pos'])
            cv_pos_recall = nltk.recall(refsets['pos'], testsets['pos'])
            cv_pos_fmeasure = nltk.f_measure(refsets['pos'], testsets['pos'])
            cv_neg_precision = nltk.precision(refsets['neg'], testsets['neg'])
            cv_neg_recall = nltk.recall(refsets['neg'], testsets['neg'])
            cv_neg_fmeasure =  nltk.f_measure(refsets['neg'], testsets['neg'])
                    
            accuracy.append(cv_accuracy)
            pos_precision.append(cv_pos_precision)
            pos_recall.append(cv_pos_recall)
            neg_precision.append(cv_neg_precision)
            neg_recall.append(cv_neg_recall)
            pos_fmeasure.append(cv_pos_fmeasure)
            neg_fmeasure.append(cv_neg_fmeasure)
            
            cv_count += 1
                
        print('---------------------------------------')
        print('N-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')')
        print('---------------------------------------')
        print('accuracy:', sum(accuracy) / n)
        print('precision', (sum(pos_precision)/n + sum(neg_precision)/n) / 2)
        print('recall', (sum(pos_recall)/n + sum(neg_recall)/n) / 2)
        print('f-measure', (sum(pos_fmeasure)/n + sum(neg_fmeasure)/n) / 2)
        if cl == 'maxent':
            maxent_accuracy_next = (sum(accuracy) / n)
            maxent_accuracy.append(maxent_accuracy_next)
        elif cl == 'svm':
            svm_accuracy_next = (sum(accuracy) / n)
            svm_accuracy.append(svm_accuracy_next)
        else:
            nb_accuracy_next = (sum(accuracy) / n)
            nb_accuracy.append(nb_accuracy_next)
        
        
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
#print (word_scores)
#raw_input('>')



#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

print "----------Unigram features-----------"
evaluate_classifier(word_feats)
#raw_input('>')
#evaluate_classifier(stopword_filtered_word_feats)
#raw_input('>')
print "----------Bigram features----------"
evaluate_classifier(bigram_word_feats)
#raw_input('>')
print "----------Trigram features----------"
evaluate_classifier(trigram_word_feats)    
#raw_input('>')
#evaluate_classifier(bigram_word_feats_stopwords)
#raw_input('>')
#numbers of features to select

def plot_accuracy_curve(maxent_accuracy, svm_accuracy, nb_accuracy , numbers_to_test):

    plt.figure()
    title = 'Features vs Accuracy'
    plt.title(title)
    ylim=(0.3, 1.01)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("No of features used")
    plt.ylabel("Accuracy Score")
    train_sizes = numbers_to_test
    train_scores_nb = nb_accuracy
    train_scores_svm = svm_accuracy   
    train_scores_maxent = maxent_accuracy   
    
    plt.grid()

    
    plt.plot(train_sizes, train_scores_nb, '.--', color="r",
             label="Naive bayes Training score")
    plt.plot(train_sizes, train_scores_svm, 'v-', color="y",
             label="SVM Training score")
    plt.plot(train_sizes, train_scores_maxent, '^-.', color="b",
             label="Maximum Entropy Training score")

    plt.legend(loc="best")
    return plt



#numbers_to_test = [10, 100, 1000, 10000, 25000]
#numbers_to_test = np.linspace(1000, 100000, 4)
#numbers_to_test = numbers_to_test.astype(int)
#tries the best_word_features mechanism with each of the numbers_to_test of features
#for num in numbers_to_test:
#	print 'evaluating best %d word features' % (num)
#	best_words = find_best_words(word_scores, num)
#	evaluate_classifier(best_word_features)
	
	
#print maxent_accuracy
#print svm_accuracy
#print nb_accuracy



#plot_accuracy_curve(maxent_accuracy[4:], svm_accuracy[4:], nb_accuracy[4:], numbers_to_test)
#plt.show()
