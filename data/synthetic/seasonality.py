import numpy as np

def retail_seasonality(n_weeks: int) -> np.ndarray:
    """
    Returns a multiplicative seasonal index of length n_weeks.
    Retail pattern: peaks at weeks 47-52 (Holiday) and 13-14 (Spring).
    """
    seasonality = np.ones(n_weeks, dtype=float)
    
    for i in range(n_weeks):
        woy = (i % 52) + 1  # 1-indexed week of year
        if (47 <= woy <= 52) or (13 <= woy <= 14):
            seasonality[i] = 1.35  # Peak multiplier
            
    return seasonality

def b2b_seasonality(n_weeks: int) -> np.ndarray:
    """
    Returns a multiplicative seasonal index of length n_weeks.
    B2B pattern: peaks at Q1 start (weeks 1-4) and Q3 start (weeks 27-30).
    """
    seasonality = np.ones(n_weeks, dtype=float)
    
    for i in range(n_weeks):
        woy = (i % 52) + 1
        if (1 <= woy <= 4) or (27 <= woy <= 30):
            seasonality[i] = 1.25  # Peak multiplier
            
    return seasonality

def flat_seasonality(n_weeks: int) -> np.ndarray:
    """
    Returns a flat, all-ones seasonal index of length n_weeks.
    """
    return np.ones(n_weeks, dtype=float)

def uniform_seasonality(n_weeks) -> np.ndarray:
    """Truly flat — no seasonal variation, just ones."""
    return np.ones(n_weeks)

def event_driven_seasonality(n_weeks: int) -> np.ndarray:
    """Sharp spikes at specific weeks simulating product launches or events."""
    index = np.ones(n_weeks)
    # Two major event spikes per year
    event_weeks = [8, 12, 60, 64]  # e.g. trade show seasons
    for w in event_weeks:
        if w < n_weeks:
            index[w] *= 2.5
        if w + 1 < n_weeks:
            index[w + 1] *= 1.8
        if w - 1 >= 0:
            index[w - 1] *= 1.4
    return index
