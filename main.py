# This is a sample Python script.
from LeetFunctions import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    str = 'asdf'
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
    print(trie.startsWith('as'))