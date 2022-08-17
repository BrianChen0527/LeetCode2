# This is a sample Python script.
import bisect
import collections

from LeetFunctions import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys

# import numpy as np
# import pandas as pd
# from sklearn import ...
from LeetFunctions import WordDictionary
import random


def numDigits(num):
    return int(math.log10(num))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    weights = [1, 2, 3]
    values = [10, 15, 40]
    print(knapsackClassic(weights,values, 6))
