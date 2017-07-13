"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

N = 3
men_means = (80.04, 81.26, 80.85)
#men_std = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.20       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='r')

women_means = (79.02, 80.44, 80.04)
#women_std = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, women_means, width, color='y')

women_means2 = (77.80, 80.85, 77.59)
#women_std = (3, 5, 2, 3, 3)
rects3 = ax.bar(ind + width*2, women_means2, width, color='g')

women_means3 = (82.28, 82.69, 82.48)
#women_std = (3, 5, 2, 3, 3)
rects4 = ax.bar(ind + width*3, women_means3, width, color='b')


# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by Features and accuracy')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Unigram', 'Bigram', 'trigram'))
plt.ylim([75,85])

ax.legend((rects1[0], rects2[0], rects3[0],rects4[0]), ('Naive Bayes', 'SVM', 'ME', 'Ensemble'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.00*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.show()

