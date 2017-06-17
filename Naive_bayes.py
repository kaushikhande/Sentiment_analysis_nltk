# A nice python class that lets you count how many times items occur in a list
from collections import Counter
import csv
import re

# Read in the training data.
with open("review.csv", 'r') as file:
  reviews = list(csv.reader(file))
  
for r in reviews:
	if r[1] == 'positive':
		r[1] = '1'
	else:
		r[1] = '-1'
		
		

def get_text(reviews, score):
  # Join together the text in the reviews for a particular tone.
  # We lowercase to avoid "Not" and "not" being seen as different words, for example.
  return " ".join([r[0].lower() for r in reviews if r[1] == str(score)])

def count_text(text):
  # Split text into words based on whitespace.  Simple but effective.
  words = re.split("\s+", text)
  # Count up the occurence of each word.
  return Counter(words)

negative_text = get_text(reviews, -1)
positive_text = get_text(reviews, 1)
# Generate word counts for negative tone.
negative_counts = count_text(negative_text)
#print negative_counts
# Generate word counts for positive tone.
positive_counts = count_text(positive_text)
#print positive_counts

print("Negative text sample: {0}".format(negative_text[:120]))
print("Positive text sample: {0}".format(positive_text[:100]))

import re
from collections import Counter

def get_y_count(score):
  # Compute the count of each classification occuring in the data.
  return len([r for r in reviews if r[1] == str(score)])

# We need these counts to use for smoothing when computing the prediction.
positive_review_count = get_y_count(1)
negative_review_count = get_y_count(-1)

# These are the class probabilities (we saw them in the formula as P(y)).
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)

def make_class_prediction(text, counts, class_prob, class_count):
  prediction = 1
  text_counts = Counter(re.split("\s+", text))
  for word in text_counts:
      # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
      # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
      # We also smooth the denominator counts to keep things even.
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Now we multiply by the probability of the class existing in the documents.
  return prediction * class_prob

# As you can see, we can now generate probabilities for which class a given review is part of.
# The probabilities themselves aren't very useful -- we make our classification decision based on which value is greater.
print("Review: {0}".format(reviews[0][0]))
print("Negative prediction: {0}".format(make_class_prediction(reviews[0][0], negative_counts, prob_negative, negative_review_count)))
print("Positive prediction: {0}".format(make_class_prediction(reviews[0][0], positive_counts, prob_positive, positive_review_count)))

#import csv

def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
      return -1
    return 1

#with open("review.csv", 'r') as file:
#    test = list(csv.reader(file))

test = reviews
#print(test)
predictions = [make_decision(r[0], make_class_prediction) for r in test]
print(predictions)
actual = [int(r[1]) for r in test]
print(actual)

from sklearn import metrics

# Generate the roc curve using scikits-learn.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))
