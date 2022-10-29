import numpy as np
from scipy import stats


import logging
import math
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six import string_types
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix, f1_score, SCORERS

dev_essay_scores = [12, 8, 3, 8, 9, 8, 8, 10, 7, 10, 8, 10, 8, 10, 6, 7, 10, 10, 7, 10, 9, 10, 6, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 10, 8, 8, 9, 6, 10, 8, 9, 6, 9, 7, 6, 10, 8, 10, 8, 10, 10, 11, 10, 8, 9, 12, 10, 8, 9, 6, 10, 9, 8, 7, 6, 8, 9, 6, 6, 8, 9, 9, 7, 8, 8, 11, 9, 8, 8, 8, 7, 8, 8, 12, 7, 8, 8, 8, 7, 10, 9, 8, 8, 8, 8, 7, 8, 10, 2, 8, 9, 8, 8, 11, 9, 8, 5, 7, 8, 9, 7, 10, 8, 7, 8, 8, 11, 8, 8, 9, 7, 8, 8, 10, 11, 8, 8, 8, 9, 6, 8, 6, 12, 10, 8, 8, 10, 10, 6, 10, 8, 9, 10, 10, 7, 9, 8, 8, 8, 8, 8, 9, 8, 2, 10, 8, 8, 10, 10, 12, 8, 12, 8, 9, 7, 11, 9, 10, 8, 8, 10, 8, 10, 9, 6, 8, 7, 8, 9, 8, 11, 9, 11, 8, 7, 8, 8, 8, 6, 9, 11, 10, 8, 8, 8, 10, 9, 9, 8, 8, 9, 7, 8, 12, 8, 10, 9, 8, 8, 10, 10, 6, 6, 8, 8, 12, 9, 8, 8, 8, 8, 10, 11, 8, 11, 9, 9, 10, 8, 11, 10, 9, 8, 10, 8, 9, 6, 10, 10, 8, 7, 11, 7, 11, 11, 9, 7, 10, 9, 8, 10, 8, 7, 9, 9, 8, 8, 8, 10, 9, 8, 8, 11, 10, 8, 6, 12, 8, 9, 9, 9, 9, 7, 8, 10, 8, 8, 8, 9, 10, 12, 9, 10, 6, 9, 6, 11, 8, 9, 8, 9, 9, 7, 8, 8, 10, 6, 6, 8, 8, 8, 8, 6, 6, 8, 8, 7, 8, 8, 11, 8, 11, 8, 8, 9, 11, 8, 8, 7, 8, 8, 8, 11, 8, 9, 9, 8, 9, 10, 9, 8, 8, 8, 8, 10, 9, 10, 8, 6, 9, 9, 9, 9, 8, 8, 11, 8, 8, 8, 12, 11, 8, 8, 2, 7]


all_pred_scores = [10, 8, 5, 8, 8, 8, 8, 10, 7, 9, 9, 10, 8, 10, 8, 8, 11, 9, 6, 10, 9, 9, 6, 8, 8, 10, 8, 8, 8, 8, 8, 8, 8, 8, 10, 8, 8, 8, 6, 8, 9, 10, 6, 8, 8, 7, 11, 8, 10, 9, 10, 9, 10, 9, 8, 8, 10, 12, 8, 8, 6, 9, 10, 8, 6, 6, 8, 9, 6, 8, 8, 9, 10, 8, 9, 8, 10, 9, 9, 8, 8, 8, 6, 8, 10, 8, 8, 8, 8, 8, 9, 8, 8, 9, 8, 8, 8, 8, 10, 2, 9, 9, 8, 8, 12, 9, 8, 4, 6, 8, 8, 8, 10, 9, 6, 8, 8, 10, 8, 9, 9, 8, 8, 8, 9, 10, 9, 8, 8, 9, 7, 8, 7, 12, 10, 6, 8, 10, 10, 6, 11, 10, 9, 10, 10, 7, 8, 8, 8, 9, 8, 8, 8, 8, 2, 10, 8, 8, 9, 10, 10, 8, 12, 8, 9, 8, 10, 8, 10, 8, 8, 10, 9, 10, 8, 6, 8, 8, 9, 9, 8, 9, 8, 12, 8, 8, 8, 8, 8, 6, 8, 11, 10, 9, 7, 8, 10, 9, 8, 8, 8, 10, 8, 8, 12, 8, 10, 8, 8, 8, 10, 8, 8, 8, 6, 8, 10, 9, 8, 8, 8, 8, 9, 9, 9, 8, 8, 9, 10, 9, 9, 9, 8, 10, 9, 8, 8, 8, 10, 8, 8, 8, 10, 8, 10, 10, 8, 8, 10, 8, 9, 10, 8, 8, 11, 8, 6, 8, 8, 10, 8, 8, 8, 9, 10, 8, 4, 10, 8, 10, 8, 9, 8, 8, 8, 9, 10, 8, 8, 8, 8, 11, 10, 9, 6, 9, 6, 12, 8, 8, 8, 9, 7, 8, 8, 8, 9, 6, 8, 8, 8, 8, 8, 6, 6, 8, 8, 8, 8, 8, 10, 8, 10, 8, 8, 9, 12, 8, 8, 8, 8, 9, 8, 12, 8, 8, 9, 8, 10, 8, 8, 9, 8, 8, 8, 11, 9, 10, 8, 6, 8, 9, 9, 10, 8, 8, 10, 8, 6, 8, 10, 10, 8, 9, 2, 7]


dev_essay_scores = [12, 8, 3, 8, 9, 8, 8, 10, 7, 10, 8, 10, 8, 10, 6, 7, 10, 10, 7, 10, 9, 10, 6, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 10, 8, 8, 9, 6, 10, 8, 9, 6, 9, 7, 6, 10, 8, 10, 8, 10, 10, 11, 10, 8, 9, 12, 10, 8, 9, 6, 10, 9, 8, 7, 6, 8, 9, 6, 6, 8, 9, 9, 7, 8, 8, 11, 9, 8, 8, 8, 7, 8, 8, 12, 7, 8, 8, 8, 7, 10, 9, 8, 8, 8, 8, 7, 8, 10, 2, 8, 9, 8, 8, 11, 9, 8, 5, 7, 8, 9, 7, 10, 8, 7, 8, 8, 11, 8, 8, 9, 7, 8, 8, 10, 11, 8, 8, 8, 9, 6, 8, 6, 12, 10, 8, 8, 10, 10, 6, 10, 8, 9, 10, 10, 7, 9, 8, 8, 8, 8, 8, 9, 8, 2, 10, 8, 8, 10, 10, 12, 8, 12, 8, 9, 7, 11, 9, 10, 8, 8, 10, 8, 10, 9, 6, 8, 7, 8, 9, 8, 11, 9, 11, 8, 7, 8, 8, 8, 6, 9, 11, 10, 8, 8, 8, 10, 9, 9, 8, 8, 9, 7, 8, 12, 8, 10, 9, 8, 8, 10, 10, 6, 6, 8, 8, 12, 9, 8, 8, 8, 8, 10, 11, 8, 11, 9, 9, 10, 8, 11, 10, 9, 8, 10, 8, 9, 6, 10, 10, 8, 7, 11, 7, 11, 11, 9, 7, 10, 9, 8, 10, 8, 7, 9, 9, 8, 8, 8, 10, 9, 8, 8, 11, 10, 8, 6, 12, 8, 9, 9, 9, 9, 7, 8, 10, 8, 8, 8, 9, 10, 12, 9, 10, 6, 9, 6, 11, 8, 9, 8, 9, 9, 7, 8, 8, 10, 6, 6, 8, 8, 8, 8, 6, 6, 8, 8, 7, 8, 8, 11, 8, 11, 8, 8, 9, 11, 8, 8, 7, 8, 8, 8, 11, 8, 9, 9, 8, 9, 10, 9, 8, 8, 8, 8, 10, 9, 10, 8, 6, 9, 9, 9, 9, 8, 8, 11, 8, 8, 8, 12, 11, 8, 8, 2, 7]
all_pred_scores = [10, 8, 5, 8, 8, 8, 8, 10, 7, 9, 9, 10, 8, 10, 8, 8, 11, 9, 6, 10, 9, 9, 6, 8, 8, 10, 8, 8, 8, 8, 8, 8, 8, 8, 10, 8, 8, 8, 6, 8, 9, 10, 6, 8, 8, 7, 11, 8, 10, 9, 10, 9, 10, 9, 8, 8, 10, 12, 8, 8, 6, 9, 10, 8, 6, 6, 8, 9, 6, 8, 8, 9, 10, 8, 9, 8, 10, 9, 9, 8, 8, 8, 6, 8, 10, 8, 8, 8, 8, 8, 9, 8, 8, 9, 8, 8, 8, 8, 10, 2, 9, 9, 8, 8, 12, 9, 8, 4, 6, 8, 8, 8, 10, 9, 6, 8, 8, 10, 8, 9, 9, 8, 8, 8, 9, 10, 9, 8, 8, 9, 7, 8, 7, 12, 10, 6, 8, 10, 10, 6, 11, 10, 9, 10, 10, 7, 8, 8, 8, 9, 8, 8, 8, 8, 2, 10, 8, 8, 9, 10, 10, 8, 12, 8, 9, 8, 10, 8, 10, 8, 8, 10, 9, 10, 8, 6, 8, 8, 9, 9, 8, 9, 8, 12, 8, 8, 8, 8, 8, 6, 8, 11, 10, 9, 7, 8, 10, 9, 8, 8, 8, 10, 8, 8, 12, 8, 10, 8, 8, 8, 10, 8, 8, 8, 6, 8, 10, 9, 8, 8, 8, 8, 9, 9, 9, 8, 8, 9, 10, 9, 9, 9, 8, 10, 9, 8, 8, 8, 10, 8, 8, 8, 10, 8, 10, 10, 8, 8, 10, 8, 9, 10, 8, 8, 11, 8, 6, 8, 8, 10, 8, 8, 8, 9, 10, 8, 4, 10, 8, 10, 8, 9, 8, 8, 8, 9, 10, 8, 8, 8, 8, 11, 10, 9, 6, 9, 6, 12, 8, 8, 8, 9, 7, 8, 8, 8, 9, 6, 8, 8, 8, 8, 8, 6, 6, 8, 8, 8, 8, 8, 10, 8, 10, 8, 8, 9, 12, 8, 8, 8, 8, 9, 8, 12, 8, 8, 9, 8, 10, 8, 8, 9, 8, 8, 8, 11, 9, 10, 8, 6, 8, 9, 9, 10, 8, 8, 10, 8, 6, 8, 10, 10, 8, 9, 2, 7]

def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.

    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted as
                             equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert len(y_true) == len(y_pred)
    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        logger.error(
            "For kappa, the labels should be integers or strings "
            "that can be converted to ints (E.g., '4.0' or '3')."
        )
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ""
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == "linear":
                    weights[i, j] = diff
                elif wt_scheme == "quadratic":
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError(
                        "Invalid weight scheme specified for "
                        "kappa: {}".format(wt_scheme)
                    )

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[:num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[:num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= sum(sum(weights * observed)) / sum(sum(weights * expected))

    return k

dev_kappa_result = kappa(dev_essay_scores, all_pred_scores, weights='quadratic')


x= np.array(dev_essay_scores) - np.array(all_pred_scores)
print(np.mean(x), np.median(x), stats.mode(x))

print("Original essay scores:")
print(dev_essay_scores)
print("Predicted essay scores for adversarial:")
print(all_pred_scores)
unique_elements, counts_elements = np.unique(dev_essay_scores, return_counts=True)
print("Frequency of unique values of the original scores")
print(np.asarray((unique_elements, counts_elements)))
unique_elements, counts_elements = np.unique(all_pred_scores, return_counts=True)
print("Frequency of unique values of the predicted scores")
print(np.asarray((unique_elements, counts_elements)))            
unique_elements, counts_elements = np.unique(x, return_counts=True)
print("Frequency of unique values of the difference (original - adversarial) scores")
print(np.asarray((unique_elements, counts_elements)))            
print(f"kappa result={dev_kappa_result}")            
print(f"kappa result={dev_kappa_result}")
