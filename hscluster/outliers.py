"""
A few different methods for detecting outliers.
(we are concerned with only upper outliers)
"""

import numpy as np


def outlier_threshold(values, thresh=1.5):
    """
    Detect outliers using the quartile method
    """
    q = np.percentile(values, np.arange(0, 100, 25))
    q1 = q[0]
    q3 = q[2]
    interquartile_range = q3 - q1
    outlier_thresh = q3 + (thresh * interquartile_range)

    mean = np.mean(values)
    outlier_thresh = mean if outlier_thresh == 0 else outlier_thresh
    return outlier_thresh


def jump_outlier_indices(values):
    """
    Detect outliers by the first big "jump" in values.

    E.g. given the values [1,2,3,10,12,14]

    The outliers would start from 10 on.
    """
    values = sorted(values)
    diffs = [y-x for x,y in zip(values, values[1:])]

    avg_diffs = []
    for i in range(len(diffs)):
        avg = sum(diffs[:i])/(i+1)
        avg_diffs.append(diffs[i]/(avg+1))

    return list(range(np.argmax(avg_diffs), len(values)))


def outlier_indices(values, thresh=1.5, outlier_thresh=None):
    """
    Detect outliers using the quartile method
    """
    if outlier_thresh is None:
        outlier_thresh = outlier_threshold(values, thresh=thresh)
    return [i for i, v in enumerate(values) if v > outlier_thresh]


def mad_based_outlier_indices(points, thresh=2.5):
    """
    Source: <http://stackoverflow.com/a/22357811>
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
