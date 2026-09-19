"""
this file calibrates the open set rejection threshold using only known
validation examples so the calibration never sees a real unknown image
"""

import numpy as np

def calibrate_threshold(val_known_scores,target_tpr):
    """
    this function returns the score percentile that accepts the given
    fraction of known validation examples
    """
    return float(np.percentile(val_known_scores,target_tpr*100))
