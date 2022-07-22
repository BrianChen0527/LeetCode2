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
    board = [["o", "a", "a", "n"], ["e", "t", "a", "e"], ["i", "h", "k", "r"], ["i", "f", "l", "v"]]
    words = ["oath", "pea", "eat", "rain"]

    print(findWords(board,words))
