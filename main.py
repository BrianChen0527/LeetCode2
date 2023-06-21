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
    arr1 = [[1,9,12,20,23,24,35,38],[10,21,24,31,32,34,37,38,43],[10,19,28,37],[8],[14,19],[11,17,23,31,41,43,44],[21,26,29,33],[5,11,33,41],[4,5,8,9,24,44]]
    print(numBusesToDestination(arr1, 37, 28))


if __name__ == '__main__':
    main()