
import numpy as np
import saffine.multi_detrending as md
import saffine.detrending_method as dm
from numpy import mat, shape, power, log2, multiply



## ---- Mean and standard deviation ---- ##

def get_basic_sentarc_features(arc: list[float]):
    # check if arc is empty
    if not arc:
        return None, None
    # basic features
    mean_sent = np.mean(arc)
    std_sent = np.std(arc)

    return mean_sent, std_sent

## ---- Hurst exponent ---- ##

def integrate(x: list[float]) -> np.matrix:
    return np.mat(np.cumsum(x) - np.mean(x))

def get_hurst(arc: list[float]):
    y = integrate(arc)
    uneven = y.shape[1] % 2
    if uneven:
        y = y[0, :-1]

    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)

    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    hurst = round(np.polyfit(x, y, 1)[0], 2)
    return hurst


# ## ---- Detrending ---- ##

def normalize(ts, scl01=False):
    ts = np.array(ts).flatten()

    # Return early if ts is empty or all-NaN
    if ts.size == 0 or not np.isfinite(ts).any():
        return np.array([])

    ts01 = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
    ts11 = 2 * ts01 - 1
    return ts01 if scl01 else ts11
    
# Detrending from figs
def detrend(story_arc):
    X = np.mat([float(x) for x in story_arc])
    n = len(story_arc)

    # Trim last element if needed to keep length even or consistent
    if n % 2 != 0:
        X = X[:, :-1]
        n = X.shape[1]

    # Calculate window length
    w = int(4 * np.floor(n / 20) + 1)

    # Ensure window length is odd and at least 3
    if w % 2 == 0:
        w += 1
    if w < 3:
        w = 3

    # Check window length vs data length
    if w > n:
        # window can't be bigger than data length, adjust:
        w = n if n % 2 == 1 else n - 1  # make odd and <= n
        if w < 3:
            # too small to detrend, just return normalized data
            return normalize(X).T

    for i in range(1, 2):  # reduce polynomial order to 1 instead of 2
        _, trend_ww_1 = dm.detrending_method(X, w, i)

    return normalize(trend_ww_1).T


# SINUOSITY measure (dimensionless)
# https://en.wikipedia.org/wiki/Sinuosity

# SI = channel length / straight-line (euclidean) distance
# we get a: score between 0 and 1, where 1 is a straight line

def get_sinuosity(arc):
    """
    Computes the sinuosity of a story arc.
    Returns a value between 0 and 1, where 1 indicates a straight line.
    """
    arc = list(arc)
    
    if not arc or len(arc) < 2:
        return None

    # Calculate the length of the arc
    arc_length = sum(np.sqrt(np.diff(arc)**2))

    # Calculate the straight-line distance (Euclidean distance)
    straight_line_distance = np.sqrt((arc[-1] - arc[0])**2)

    # Calculate sinuosity
    sinuosity = arc_length / straight_line_distance if straight_line_distance != 0 else 0

    return sinuosity

