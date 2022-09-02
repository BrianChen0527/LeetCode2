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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arr1 = np.array([1, 2, 3, 4], ndmin=5)
    print(arr1)
    arr3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(arr3[1:][1:])
    arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(arr3[1:, 1:])
    arr4 = np.array([1, 2, 3, 4, 5, 6, 7])
    print(arr4[-1:-5])

    # reshaping array doesnt change its base
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    print(arr.reshape(2, 4).base)

    # flatten a multi-dimensional array
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(arr.reshape(-1))

    # iterating through multi-dimensional numpy arrays
    for x in np.nditer(arr):
        print(x)
    for x in np.nditer(arr[:, ::2]):
        print(x)

    # Concatenating np arrays
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    arr = np.concatenate((arr1, arr2), axis=0)
    print(arr)
    arr = np.concatenate((arr1, arr2), axis=1)
    print(arr)

    # Stacking np arrays
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr = np.stack((arr1, arr2), axis=0)
    print(arr)
    arr = np.stack((arr1, arr2), axis=1)
    print(arr)
    newarr = arr[arr % 2 == 0]
    print(newarr)
