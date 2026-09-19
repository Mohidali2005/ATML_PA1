"""
this file runs every training stage of task four in order starting from
the cifar ten splits through the vanilla gcsc and proser checkpoints
"""

from task4.data import make_splits
from task4.methods import vanilla
from task4.methods import gcsc
from task4.methods import proser

def main():
    """
    this function calls every training stage of the task four pipeline
    one after another and prints a short label before each stage starts
    """
    print("step 1 building the cifar ten splits")
    make_splits.main()

    print("step 2 training the vanilla baseline")
    vanilla.main()

    print("step 3 training gcsc")
    gcsc.main()

    print("step 4 training proser")
    proser.main()

if __name__ == "__main__":
    main()
