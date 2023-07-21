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
    roads = [[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[0,4,5],[4,6,2]]
    countPaths(7, roads)

if __name__ == '__main__':
    main()