# This is a sample Python script.
import bisect
import collections

from LeetFunctions import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from eecs445tests import *
import sys
import re
# import numpy as np
# import pandas as pd
# from sklearn import ...
from LeetFunctions import WordDictionary
import random

import numpy as np
import pandas as pd




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv("dataset.csv")
    word_dict = extract_dictionary(df)
    arr = generate_feature_matrix(df, word_dict)
    print(word_dict)






