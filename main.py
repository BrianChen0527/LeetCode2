# functions file
import bisect
import itertools
import math
import random
from typing import List
from typing import Optional
from queue import PriorityQueue
import collections
from collections import deque
import heapq
import sys
import functools
import os
import hashlib
import subprocess
import tempfile
import shutil
import re
from LeetFunctions import *


def main():
    arr1 = ["bb","bababab","baab","abaabaa","aaba","","bbaa","aba","baa","b"]
    word = "abcddcba"
    p = PalindromePairs
    print(p.palindromePairs(p, arr1))
    n = len(word) // 2
    print(word[:n])
    print(word[n + (len(word) % 2):][::-1])
    print(p.isPalindrome(p, "baab"))

if __name__ == '__main__':
    main()