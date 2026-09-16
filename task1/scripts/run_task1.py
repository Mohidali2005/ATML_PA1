"""
This file runs every step of task one in order starting from building
the dataset splits all the way through the final analysis. Running
this one file reproduces every result used in the task one report
"""

from data import make_subset
from data import make_cue_conflicts
from scripts import extract_features
from scripts import train_heads
from analysis import evaluate_bias
from analysis import feature_similarity
from analysis import representation

def main():
    """
    This function calls every stage of the task one pipeline one after
    another and prints a short label before each stage starts
    """
    print("step 1 building dataset splits")
    make_subset.main()

    print("step 2 building cue conflict images")
    make_cue_conflicts.main()

    print("step 3 extracting backbone features")
    extract_features.main()

    print("step 4 training classifier heads")
    train_heads.main()

    print("step 5 evaluating bias and interventions")
    evaluate_bias.main()

    print("step 6 measuring feature stability")
    feature_similarity.main()

    print("step 7 fitting representation plots")
    representation.main()

if __name__ == "__main__":
    main()
