# This is a sample Python script.
from LeetFunctions import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    str1 = ""
    total = 0
    for i in range (2000,3000):
        num = str(i)
        total += num.count('2')
        str1 += num
    print(total)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
