# This is a sample Python script.
from LeetFunctions import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys

# import numpy as np
# import pandas as pd
# from sklearn import ...


def appendDict(graph, key, value):
    if key not in graph:
        graph[key] = []
    graph[key].append(value)


def findAllPaths(graph, curr, end, visited, currentRate, rate, maxRate):
    # Mark the current currency as visited and adjust currentRate
    visited.add(curr)
    currentRate = currentRate*rate
    # If current currency is same as target currency, then update maxRate
    if curr == end:
        maxRate[0] = max(float(maxRate[0]), currentRate)
    else:
        # If current vertex is not target currency
        # Recur for all the vertices adjacent to this vertex
        for exch in graph[curr]:
            currency, newRate = exch[0], float(exch[1])
            if currency not in visited:
                findAllPaths(graph, currency, end, visited, currentRate, newRate, maxRate)

    # Remove current vertex from visited and divide currentRate by the rate of previous currency in path
    visited.remove(curr)
    currentRate / rate


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    line_idx = 0
    rates = []
    graph = {}
    start, end = '', ''


    with open('test.txt') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n')
        if line_idx == 0:
            # get the rates of exchanges (and remove newLine character)
            rates = line.split(';')
            line_idx += 1

            for rate in rates:
                country1, country2, rate = rate.split(',')
                appendDict(graph, country1, (country2, float(rate)))
                appendDict(graph, country2, (country1, 1/float(rate)))

        elif line_idx == 1:
            start = line
            line_idx += 1

        elif line_idx == 2:
            end = line
            line_idx += 1

    queue = [start]
    visited = set([start])
    currRate, rate, maxRate = 1, 1, [-1.0]
    path = []
    findAllPaths(graph, start, end, visited, currRate, rate, maxRate)
    print(maxRate[0])

    '''str = 'asdf'
    trie = Trie()
    trie.insert(str)

    trie.insert('asdas')
    print(trie.search('asdf'))
    print(trie.search('asdas'))
    print(trie.search('asde'))
    print(trie.search('as'))

    print(trie.startsWith('asdf'))
    print(trie.startsWith('asdas'))
    print(trie.startsWith('asde'))
    print(trie.startsWith('as'))'''