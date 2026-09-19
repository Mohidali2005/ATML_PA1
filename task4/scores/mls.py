"""
this file scores every example by the negative of its largest raw logit
instead of a softmax probability so absolute confidence magnitude is
preserved rather than normalized away
"""

def mls_score(logits):
    """
    this function turns a batch of known class logits into an
    unknownness score equal to the negative of the maximum logit
    """
    return -logits.max(dim=1).values
