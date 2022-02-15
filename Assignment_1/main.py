import argparse
from exercise_4 import *
from exercise_6 import *
from exercise_8 import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exercise", choices=[0, 4, 6, 8], default=4, type=int,
                        help="The number of the exercise of which the code will be executed")
    args = parser.parse_args()

    if args.exercise == 4:
        exercise_4()
    elif args.exercise == 6:
        exercise_6()
    elif args.exercise == 8:
        exercise_8()
    else:
        exercise_4
        exercise_6
        exercise_8
