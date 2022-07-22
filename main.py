# This is a sample Python script.
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
    arr = []
    heapq.heappush(arr, -1)
    heapq.heappush(arr, -2)
    heapq.heappush(arr, -3)
    heapq.heappush(arr, -4)
    heapq.heapify(arr)
    print(heapq.heappop(arr))
    print(heapq.nlargest(3, [9, 1,2,3,4,5,6,7,8]))