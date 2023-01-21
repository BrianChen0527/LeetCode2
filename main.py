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

# Press the green button in the gutter to run the script.

def caesarCipher(text, key):
    ansText = ""
    for i in range(len(text)):
        ansText += chr(ord('A') + (ord(text[i]) - ord('A') + key) % 26)
    return ansText

def generatorChecker(num1, modNum):
    arr = []
    for i in range(modNum):
        mod = (num1**i) % modNum
        if mod not in arr:
            arr.append(mod)
    print(sorted(arr))

def wrapper(fn):
    print("wrapping")
    def wrapped():
        print("Start")
        fn()
        print("End")
    return wrapped

def read_sql_dump(dump):
    f = open(dump, "r")
    Lines = f.readlines()
    sql_strs = []
    sql_str = "("

    for line in Lines:
        if line in ['\n', '\r\n']:
            sql_str = sql_str[:-1]
            sql_str += ")"
            sql_strs.append(sql_str)
            sql_str = "("
        if '=' not in line:
            continue
        line = line.strip()
        if "id" in line:
            sql_str += (line[line.index(' = ') + 3:] + ',')
        else:
            sql_str += ('\'' + line[line.index(' = ') + 3:] + "\',")
    return sql_strs

if __name__ == '__main__':
    tmp = read_sql_dump("dump.txt")
    for t in tmp:
        print(t)