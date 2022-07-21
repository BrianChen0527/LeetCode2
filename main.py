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
    w = WordDictionary()

    w.addWord('abcde')
    trie = w.addWord('abcfg')
    print(trie)

    for c in 'abc':
        trie = trie[c]
    print(trie)
    newTrie = dict()
    for key in trie:
        newTrie = {**newTrie, **trie[key]}
    print(newTrie)

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