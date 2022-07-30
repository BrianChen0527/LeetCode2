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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   n = 2
   tasks = ["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"]
   c = collections.Counter(tasks)
   # numTasks = sorted(list(c.values()), key=lambda x: -x)
   dp = []
   while len(c) > 0:
      ptr = 0
      if len(dp) == ptr:
         dp += ([True] + [False] * n)

