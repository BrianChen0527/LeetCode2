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
    l1 = ListNode(1)
    l2 = ListNode(2)
    l3 = ListNode(3)
    l4 = ListNode(4)
    l5 = ListNode(5)
    l1.next = l2
    l2.next = l3
    l3.next = l4
    l4.next = l5

    l6 = reverseKGroup(l1,3)
    while l6:
        print(l6.val)
        l6 = l6.next

if __name__ == '__main__':
    main()