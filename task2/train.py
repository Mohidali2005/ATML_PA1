"""
This file runs every training stage of task two in order starting from
building the pacs splits through training every method
"""

from shared import pacs_protocol
from task2.methods import source_only
from task2.methods import dan
from task2.methods import dann
from task2.methods import cdan

def main():
    """
    This function calls every training stage of the task two pipeline
    one after another and prints a short label before each stage starts
    """
    print("step 1 building pacs splits")
    pacs_protocol.main()

    print("step 2 training source only")
    source_only.main()

    print("step 3 training dan")
    dan.main()

    print("step 4 training dann")
    dann.main()

    print("step 5 training cdan")
    cdan.main()

if __name__ == "__main__":
    main()
