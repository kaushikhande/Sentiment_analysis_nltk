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
    #print data
    count_vectorizer = CountVectorizer(binary='true',ngram_range=(1,3))
    data = count_vectorizer.fit_transform(data)
    #print data
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    #print tfidf_data
    return tfidf_data

def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    predicted_prob = classifier.predict_proba(data_test)
    predicted_prob_log = classifier.predict_log_proba(data_test)
    print predicted
    print predicted_prob
    for r in predicted_prob:
    	
    	print (r[0]/(r[0] + r[1]))*100,
    	print (r[1]/(r[0] + r[1]))*100
    #print predicted_prob_log
    #print predicted
    evaluate_model(target_test,predicted)

# read more about model evaluation metrics here
# http://scikit-learn.org/stable/modules/model_evaluation.html
def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    #print confusion_matrix(target_true,target_predicted)
    cm = confusion_matrix(target_true,target_predicted)
    print cm
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))



def main():
    data,target = load_file()
    tf_idf = preprocess()
    learn_model(tf_idf,target)


main()

