# """EDX EASE"""

# !git clone https://github.com/edx/ease.git



# # from google.colab import drive
# # drive.mount('/content/drive')

# !pip install --upgrade setuptools
# !apt-get install zlib1g-dev
# !pip install path.py --no-cache-dir



# # Commented out IPython magic to ensure Python compatibility.
# # %cd /content/ease

# !apt-get update
# !apt-get upgrade gcc

# !pwd

# !xargs -a apt-packages.txt apt-get install
# !pip install -r pre-requirements.txt

# !pip install -r requirements.txt

# !python setup.py install

# !bash -x download-nltk-corpus.sh

# !cp /content/drive/My\ Drive/custom_extractor.py /content/drive/My\ Drive/inference.py /content/ease/ease

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/ease/src/nltk
import nltk

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/ease/ease

## IMPORTS ##
from essay_set import EssaySet

from feature_extractor import FeatureExtractor
from predictor_set import PredictorSet
from predictor_extractor import PredictorExtractor
from sklearn.svm import SVR
import pickle
import pandas as pd
import csv

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve,GridSearchCV
#from sklearn import svm, grid_search

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

## TRAINING FEATURES ##

train_set = pd.read_csv('/home/mehar/github/ease/aes_data/essay6/fold_0/train.txt', sep='\t')
x = train_set.to_numpy()
tester = x.tolist()
print(len(tester))
essaylist = []
scorelist = []
for i in range(0, len(tester)):
    z = tester[i]
    y = z[0].split(', ', 1)
    #print(y)
    scorelist.append(float(y[0]))
    essaylist.append(y[1])

train = EssaySet()
print("Done1")
for i in range(0, len(essaylist)):
    train.add_essay(essaylist[i], scorelist[i])

print("Done2")
features=FeatureExtractor()
features.initialize_dictionaries(train)
X = features.gen_feats(train)
# print("features train", X)
print("Done3")


## TESTING FEATURES ##

test_set = pd.read_csv('/home/mehar/github/ease/aes_data/essay6/fold_0/test.txt', sep='\t')
x = test_set.to_numpy()
tester = x.tolist()
print(len(tester))
test_scorelist = []
test_essaylist = []

for i in range(0, len(tester)):
    z = tester[i]
    y = z[0].split(', ', 1)
    test_scorelist.append(float(y[0]))
    test_essaylist.append(y[1])
count = 0

test = EssaySet(essaytype="test")
for i in range(0, len(test_essaylist)):
    test.add_essay(test_essaylist[i], test_scorelist[i])
Y = features.gen_feats(test)
# print("features test", Y)
print("Done4")

## SCALING
scaled_train = []
for i in range(0, len(scorelist)):
    scaled_train.append(float((np.clip((scorelist[i]), a_min=0, a_max=4)/4)))

print("start training and prediciton")

# print(scaled_train)
from sklearn.svm import SVR
## TRAINING & PREDICTING
# print("train and predict")
# param = {'kernel' : ('linear'),'C' : [1,5,10]},
# parameters = {'kernel':('linear'), 'C':[1.5, 10], 'epsilon':[0.1, 0.2, 0.5, 0.3]}
# svr = SVR()
# clf = GridSearchCV(svr, param)
# clf.fit(X, scaled_train)
# print(clf.best_params_)

# clf = SVR(C=10, epsilon=0.1,kernel='linear')
# print("done7")
# clf.fit(X, scaled_train)
# print("done5")

Cs = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
gammas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# gammas = [0.0000001, 0.0000010.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

param_grid = {'C': Cs, 'gamma' : gammas}
clf = GridSearchCV(SVR(kernel='rbf'), param_grid, cv =5)
clf.fit(X, scaled_train)
print(clf.best_params_)
# clf = SVR(C=10, gamma = 0.001, kernel='rbf') 
# clf.fit(X, scaled_train)

final = clf.predict(Y)
# print(final)
# INVERSE_SCALING
finals = np.rint(np.clip(final,a_min=0,a_max=1)*4)
print("done6")
## QWK Score
print(quadratic_weighted_kappa(test_scorelist,finals)) ## test_scorelist   babble_score    gpt_score
