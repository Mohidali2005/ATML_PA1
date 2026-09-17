"""
This file runs every training stage of task three in order starting
from the source splits already built by task two through the erm dan
dg and sam checkpoints
"""

from shared import pacs_protocol
from task3.methods import erm
from task3.methods import dan_dg
from task3.methods import sam

def main():
    """
    This function calls every training stage of the task three pipeline
    one after another and prints a short label before each stage starts
    """
    print("step 1 building pacs splits")
    pacs_protocol.main()

    print("step 2 loading the erm checkpoint from task two")
    erm.main()

    print("step 3 training dan dg")
    dan_dg.main()

    print("step 4 training sam")
    sam.main()

if __name__ == "__main__":
    main()
