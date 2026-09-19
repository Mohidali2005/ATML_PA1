"""
this file runs the full task four evaluation pipeline once every model
is trained and its outputs are cached, producing every required table
and figure in one pass
"""

from task4 import extract_outputs
from task4.evaluation import score_comparison
from task4.evaluation import method_comparison
from task4.evaluation import failure_analysis
from task4.evaluation import tsne_features

def main():
    """
    this function extracts and caches every model's outputs, then runs
    the post hoc score comparison, the method comparison, the failure
    analysis, and the feature space visualization in order
    """
    print("step 1 extracting and caching model outputs")
    extract_outputs.main()

    print("step 2 comparing post hoc scores on the vanilla model")
    score_comparison.main()

    print("step 3 comparing vanilla gcsc and proser")
    method_comparison.main()

    print("step 4 running the failure analysis")
    failure_analysis.main()

    print("step 5 visualizing the feature space")
    tsne_features.main()

if __name__ == "__main__":
    main()
