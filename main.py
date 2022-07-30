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
   cache = collections.OrderedDict()
   cache[1] = 100
   cache[2] = 200
   cache.move_to_end(1)
   print(cache)
   cache.popitem(last=False)
   print(cache)
