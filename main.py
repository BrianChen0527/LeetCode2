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




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n1 = customNode(1, 200)
    n2 = customNode(2, 130)
    n3 = customNode(3, 450)
    n4 = customNode(4, 400)
    n5 = customNode(5, 100)
    arr = []

    n1.neighbors = [n2, n3]
    n2.neighbors = [n4]
    n4.neighbors = [n5]
    n3.neighbors = [n5]
    print(shortestDist(n1, n5))

