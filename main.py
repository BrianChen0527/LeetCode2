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


class customNode:
    def __init__(self, val: int, dist: int):
        self.val = val
        self.dist = dist
        self.neighbors = []


def shortestDist1(start: customNode, end: customNode):
    currTree = set()

    canReach = [(0, start)]
    while canReach:
        currDist, currNode = heapq.heappop(canReach)
        if currNode.val in currTree:
            continue
        if currNode == end:
            return currDist

        currTree.add(currNode.val)
        for n in currNode.neighbors:
            heapq.heappush(canReach, (currDist + n.dist, n))

# https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/submissions/
def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
    def childrenDFS(node):
        if node in visited:
            return visited[node]

        children = set()
        for v in G[node]:
            children.add(v)
            children = children | childrenDFS(v)

        visited[node] = children
        return children

    G = collections.defaultdict(list)
    for e in edges:
        G[e[1]].append(e[0])

    ans, visited = [[] for _ in range(n)], {}
    for i in range(n):
        ans[i] = sorted(list(childrenDFS(i)))
    return ans


def shortestDist2(start: customNode, end: customNode, nodesGraph: dict):
    currTree = set()

    canReach = [(0, start)]
    while canReach:
        currDist, currNode = heapq.heappop(canReach)
        if currNode in currTree:
            continue
        if currNode == end:
            return currDist

        currTree.add(currNode)
        for n in nodesGraph[currNode]:
            heapq.heappush(canReach, (currDist + n[1], n[0]))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n1 = customNode(1, 200)
    n2 = customNode(2, 130)
    n3 = customNode(3, 450)
    n4 = customNode(4, 400)
    n5 = customNode(5, 100)
    arr = []

    n1.neighbors = [n2, n3]
    n2.neighbors = [n4]
    n4.neighbors = [n5]
    n3.neighbors = [n5]
    print(shortestDist(n1, n5))

