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

import numpy as np


def minProductSubarray(arr):
    minProduct = arr[0]
    currProduct = arr[0]
    for i in range(len(arr)):
        currProduct = min(currProduct * arr[i], arr[i])
        minProduct = min(minProduct, currProduct)
    return minProduct

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    s = 'rjaljr'
    print(longestPalindromicSubsequence(s))
