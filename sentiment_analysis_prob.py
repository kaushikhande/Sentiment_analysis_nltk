import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
#import numpy as np
# review.csv contains two columns
# first column is the review content (quoted)
# second column is the assigned sentiment (positive or negative)

con_rev = []
cross_ref = []

data1 =[]
def load_file():
    with open('review.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            # skip missing data
            if row[0] and row[1]:
                data.append(row[0])
                #print row[0]
                target.append(row[1])
                #print row[1]
                #raw_input('>')

        return data,target

# preprocess creates the term frequency matrix for the review data set
def preprocess():
    data,target = load_file()
    
    count_vectorizer = CountVectorizer(ngram_range=(1,1))
    
    data = count_vectorizer.fit_transform(data)
    data1 = ["This book is interesting","It is boring and interesting interesting"]
    print data1[0]
    print data1[1]
    data1 = count_vectorizer.fit_transform(data1)
    print data1
    raw_input('>')

    #print data
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    tfidf_data1 = TfidfTransformer(use_idf=False).fit_transform(data1)
    print tfidf_data1
    raw_input('>')
    return tfidf_data



predicted_values = []
def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    
    classifier = BernoulliNB().fit(data_train,target_train)
    print data_test[0]
    raw_input('>')
    predicted = classifier.predict(data_test)
    print data1
    #predicted1 = classifier.predict(data1)
    #raw_input('--')
    #print data1
    #print predicted1
    for i in predicted:
    	cross_ref.append(i)
    #print "cross_ref"
    #print cross_ref
    #print cross_ref
    print np.shape(cross_ref)
    
    predicted_prob = classifier.predict_proba(data_test)
    for i in predicted_prob:
        predicted_values.append(i)
    #raw_input('>')
    predicted_prob_log = classifier.predict_log_proba(data_test)
    #print data_test
    
    w = len(predicted)
    #print w
    h = 2
    i = 0
    matrix = [[0 for x in range(h)]for y in range(w)]
    #print np.shape(matrix)
    for p in predicted:
    	matrix[i][0] = p
    	i = i+1
    
    i = 0
    for p in target_test:
    	matrix[i][1] = p
    	i = i+1
    	
    	
    i = 0
    for p in matrix:
    	print i,
    	print matrix[i][0],
    	print matrix[i][1]
    	i = i + 1
    	
    
    i = 0
    for p in matrix:
    	if matrix[i][0] != matrix[i][1]:
    		con_rev.append(i)
    		
    	i = i + 1
    	
    	
   
   
   
    #print np.shape(data_test)
    	
    
    #print predicted, target_test
    #print predicted_prob
    #for r in predicted_prob:	
    #	print (r[0]/(r[0] + r[1]))*100,
    #	print (r[1]/(r[0] + r[1]))*100
    #print predicted_prob_log
    #print predicted
    #sample = classifier.predict("This book is interesting")
    #print sample
    #raw_input(">")
    evaluate_model(target_test,predicted)

# read more about model evaluation metrics here
# http://scikit-learn.org/stable/modules/model_evaluation.html
def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    #print confusion_matrix(target_true,target_predicted)
    cm = confusion_matrix(target_true,target_predicted)
    print cm
    raw_input('>')
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))
    



def main():
    data,target = load_file()
    tf_idf = preprocess()
    learn_model(tf_idf,target)
    #print cross_ref
    data1, target1 = load_file()
    data_train1,data_test1,target_train1,target_test1 = cross_validation.train_test_split(data1,target1,test_size=0.20,random_state=43)
    print np.shape(data_test1)
    for i in range(0,75):
    	print "==========================="
    	print data_test1[i]        # ########## test reviews from 0 to range values
    	print "Actual Label ->",
    	print target_test1[i]      ############## correct labeled values for respective test review
    	print "Predicted Label ->",
    	print cross_ref[i]     ########### predicted label for respective test review
    	print predicted_values[i]  ############ predicted values [ negative %, positive %]
    	
    #print "cross_ref"
    #print cross_ref[0:6]
	

main()

