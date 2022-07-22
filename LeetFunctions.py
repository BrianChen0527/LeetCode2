# functions file
import math
from typing import List
from typing import Optional
from queue import PriorityQueue
import collections
from collections import deque
import heapq
import sys


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_sub_array_of_size_k(k, arr):
    max_sum = 0
    window_sum = 0
    for i in range(len(arr)):
        window_sum += arr[i]
        if i >= k:
            window_sum -= arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum


def find_averages_of_subarrays(K, arr):
    sums = []
    window_sum, start_pos = 0.0, 0

    for i in range(len(arr)):
        window_sum += arr[i]
        if i - start_pos + 1 == K:
            sums.append(window_sum / 5)
            start_pos += 1
            window_sum -= arr[start_pos]
    return sums


def smallest_subarray_with_given_sum(s, arr):
    start = 0
    window_sum = 0
    min_length = 1000
    for i in range(len(arr)):
        window_sum += arr[i]
        if window_sum >= s:
            while window_sum - arr[start] >= s:
                window_sum -= arr[start]
                start += 1
            window_length = i - start + 1
            min_length = min(min_length, window_length)
    return min_length


# Given a string, find the length of the longest substring in it with no more than K distinct characters.
def longest_substring_with_k_distinct(str1, k):
    chars_freq = {}
    start = 0
    max_len = -1
    for j in range(len(str1)):
        letter = str1[j]
        if letter not in chars_freq:
            chars_freq[letter] = 0
        chars_freq[letter] += 1

        while len(chars_freq) > k:
            chars_freq[str1[start]] -= 1
            if chars_freq[str1[start]] == 0:
                del chars_freq[str1[start]]
            start += 1

        max_len = max(max_len, j - start + 1)
    return max_len


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def print_interval(self):
        print("[" + str(self.start) + ", " + str(self.end) + "]", end='')


# Given a list of intervals, merge all the overlapping intervals to produce a list that has only mutually exclusive
# intervals.
def merge(intervals):
    mergedIntervals = []

    if len(intervals) < 2:
        return intervals

    intervals = sorted(intervals, key=lambda interval: interval.start)
    start = intervals[0].start
    end = intervals[0].end

    for i in range(1, len(intervals)):
        if intervals[i].start < end:
            end = max(end, intervals[i].end)
        else:
            mergedIntervals.append(Interval(start, end))
            start = intervals[i].start
            end = intervals[i].end
    mergedIntervals.append(Interval(start, end))
    return mergedIntervals


# reverse a string str
def reverse_string(str):
    reversed_str = ""
    for i in range(len(str)):
        reversed_str += str[len(str) - 1 - i]
    return reversed_str


# Merge nums1 and nums2 into a single array sorted in non-decreasing order BY ALTERING THE FIRST GIVEN ARRAY
# https://leetcode.com/problems/merge-sorted-array/
def merge_sorted_arrays(nums1, m, nums2, n):
    new_arr = [None] * (m + n)
    c1, c2 = 0, 0
    for i in range(m + n):
        if c1 < m or c2 >= n or nums1[c1] < nums2[c2]:
            new_arr[i] = nums1[c1]
            c1 += 1
        else:
            new_arr[i] = nums2[c2]
            c2 += 1


def merge_sorted_arrays2(nums1, m, nums2, n):
    c1, c2 = m - 1, n - 1
    for i in reversed(range(m + n)):
        if c1 >= 0 and (c2 < 0 or nums1[c1] > nums2[c2]):
            nums1[i] = nums1[c1]
            c1 -= 1
        else:
            nums1[i] = nums2[c2]
            c2 -= 1


def firstRecurringNumber(arr):
    hashmap = {}
    for i in range(len(arr)):
        if (hashmap.get(arr[i])):
            return arr[i]
        hashmap[arr[i]] = 1
    return -1


# Given the head of a singly linked list, reverse the list, and return the reversed list.
# https://leetcode.com/problems/reverse-linked-list/
def reverseList(self, head):
    prevNode = None
    while head:
        nextNode = head.next
        head.next = prevNode
        prevNode = head
        head = nextNode
    return prevNode


# Given a string s, return the longest palindromic substring in s.
def longestPalindrome(self, s):
    p = s[0]
    for i in range(1, len(s)):
        if (i + 1 < len(s) and s[i - 1] == s[i + 1]):
            temp = self.helper(s, i, i)
            if (len(temp) > len(p)):
                p = temp
        if (s[i - 1] == s[i]):
            temp = self.helper(s, i - 1, i)
            if (len(temp) > len(p)):
                p = temp
    return p


# helper function for longestPalindrome
def helper(self, s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1
    return s[l + 1:r]


# https://leetcode.com/problems/two-sum/submissions/
def twoSum(self, nums, target: int):
    hash = {}
    for i in range(len(nums)):
        complement = (target - nums[i])
        if complement in hash:
            return [hash[complement], i]
        hash[nums[i]] = i


def maxProfit(self, prices):
    maxP, maxTrade = 0, 0
    for i in reversed(prices):
        if i > maxTrade:
            maxTrade = i
        if maxTrade - i > maxP:
            maxP = maxTrade - i
    return maxP


# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/submissions/
def findMin(self, nums):
    l, r = 0, len(nums) - 1
    mid = (l + r) // 2
    while l < r:
        if nums[mid] < nums[r]:
            r = mid
        else:
            l = mid + 1
        mid = (l + r) // 2
    return nums[l]


def search(nums, target):
    l, r = 0, len(nums) - 1
    mid = (l + r) // 2
    while nums[mid] != target and l < r:
        print(l, r)
        if nums[l] < nums[r]:  # no rotation
            if target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        elif nums[mid] < nums[r]:
            if target < nums[mid] or target > nums[r]:
                r = mid - 1
            else:
                l = mid + 1
        elif nums[mid] > nums[r]:
            if target > nums[mid] or target < nums[l]:
                l = mid + 1
            else:
                r = mid - 1
        mid = (l + r) // 2
    return mid if nums[mid] == target else -1


def validWordSquare(self, words: List[str]) -> bool:
    for r in range(len(words)):
        for c in range(r + 1, len(words)):
            if words[r][c] != words[c][r]:
                return False
    return True


def minDepth(self, root: Optional[TreeNode]) -> int:
    if root is None:
        return 0
    if root.left is None or root.right is None:
        return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
    else:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1


def isValidBST(root: Optional[TreeNode]) -> bool:
    def inorder(node: Optional[TreeNode], l: List[int]):
        if not node:
            return
        inorder(node.left, l)
        l.append(node.val)
        inorder(node.right, l)

    lst = []
    inorder(root, lst)
    for i in range(len(lst) - 1):
        if lst[i] >= lst[i + 1]:
            return False
    return True


def plusOne(head: ListNode) -> ListNode:
    if not head:
        return 1

    s = ""
    while head:
        s += str(head.val)
        head = head.next
    return int(s) + 1


def maxResult(nums: List[int], k: int) -> int:
    dp = [0] * len(nums)
    dp[0] = nums[0]
    deq = deque([(nums[0], 0)])
    for i in range(1, len(nums)):
        dp[i] = nums[i] + deq[0][0]

        # remove elements smaller than our new element sum
        while deq and deq[-1][0] < dp[i]:
            deq.pop()
        deq.append([dp[i], i])

        # pop_front of deq when element out of jump range
        if i - k == deq[0][1]:
            deq.popleft()
    return dp[-1]


def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
    def BSTfinder(currNode: Optional[TreeNode]) -> [int, int, int, int, bool]:
        # return format: [subtree size, largest num of subtree, smallest num of subtree, max BST subtree size, isBST]
        # base cases
        if not currNode:
            return 0, -sys.maxsize, sys.maxsize, 0, False
        if not currNode.left and not currNode.right:
            return 1, currNode.val, currNode.val, 1, True

        # our return element
        ans = [0, 0, 0, 0, 0]

        # recur through left subtree
        left = BSTfinder(currNode.left)
        # recur through right subtree
        right = BSTfinder(currNode.right)

        # Check if left + right + parent Node is a valid BST
        if left[4] and right[4] and right[2] > currNode.val > left[1]:
            ans[1] = max(right[1], currNode.val)
            ans[2] = min(left[2], currNode.val)
            ans[3] = left[3] + right[3] + 1
            ans[4] = 1
            return ans

        # If not, return max of left and right BST size
        ans[3] = max(left[3], right[3])
        ans[4] = 0
        return ans

    return BSTfinder(root)[3]


def countCornerRectangles(grid: List[List[int]]) -> int:
    # create our Dynamic Programming grid, which keeps track of occurrences in each row where col1 and col2 are both 1s
    cols = len(grid[0])
    dp = [[0 for x in range(cols)] for y in range(cols)]
    rectangles = 0
    # loop through grid
    for row in grid:
        for col1 in range(cols - 1):
            if row[col1] == 1:
                for col2 in range(col1 + 1, cols):
                    if row[col2] == 1:
                        rectangles += dp[col1][col2]
                        dp[col1][col2] += 1
    return rectangles


def minCost(costs: List[List[int]]) -> int:
    numHouses = len(costs)
    dp = [[0 for x in range(numHouses)] for y in range(3)]
    dp[0][0], dp[1][0], dp[2][0] = costs[0]

    for i in range(1, numHouses):
        prices = costs[i]
        dp[0][i] = min(dp[1][i - 1], dp[2][i - 1]) + prices[0]
        dp[1][i] = min(dp[0][i - 1], dp[2][i - 1]) + prices[1]
        dp[2][i] = min(dp[0][i - 1], dp[1][i - 1]) + prices[2]

    return min(dp[0][-1], dp[1][-1], dp[2][-1])


def countComponents(n: int, edges: List[List[int]]) -> int:
    parents = [i for i in range(n)]
    rank = [1 for i in range(n)]
    components = n

    def findParent(v: int) -> int:
        # base case
        if parents[v] == v:
            return v
        parents[v] = findParent(parents[v])
        return parents[v]

    def union(v1: int, v2: int) -> int:
        # Check if the two vertices have the same parent, if so, we have not found a new connected component
        parents[v1], parents[v2] = findParent(v1), findParent(v2)
        if parents[v1] == parents[v2]:
            return 0
        else:
            if rank[v1] > rank[v2]:
                parents[v2] = findParent(v1)
                rank[v1] += rank[v2]
            else:
                parents[v1] = findParent(v2)
                rank[v2] += rank[v1]
            return 1

    # Iterate through edges list
    for vertex1, vertex2 in edges:
        components -= union(vertex1, vertex2)
    return components


def validTree(n: int, edges: List[List[int]]) -> bool:
    visited = [False for i in range(n)]
    for n1, n2 in edges:
        visited[n1] = True
        if visited[n2]:
            return False
        visited[n2] = True
    return False not in visited


def maxNumberOfApples(weight: List[int]) -> int:
    CAPACITY = 5000
    apples = 0
    weight.sort()
    for w in weight:
        CAPACITY -= w
        if CAPACITY > 0:
            apples += 1
        else:
            break
    return apples


# https://leetcode.com/problems/same-tree/
def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if p is None and q is None:
        return True
    elif p is None or q is None:
        return False
    elif p.val == q.val:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
    else:
        return False


def minProductSum(nums1: List[int], nums2: List[int]) -> int:
    nums1.sort()
    nums2.sort(reverse=True)
    sum = 0
    for n1, n2 in zip(nums1, nums2):
        print(n1 * n2)
        sum += n1 * n2
    return sum


def wiggleSort(self, nums: List[int]) -> None:
    nums.sort()
    # Extract the smallest number if the size of nums is odd and append it to the final array
    mid = math.ceil(len(nums) / 2)
    nums[::2], nums[1::2] = nums[:mid][::-1], nums[mid:][::-1]


def arraysIntersection(arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:

    def merger(arr1: List[int], arr2: List[int]) -> List[int]:
        tmp = []
        p1, p2 = 0, 0
        while p1 < len(arr1) and p2 < len(arr2):
            if arr1[p1] < arr2[p2]:
                p1 += 1
            elif arr1[p1] > arr2[p2]:
                p2 += 1
            else:
                tmp.append(arr1[p1])
                p1, p2 = p1 + 1, p2 + 1
        return tmp

    tmp = merger(arr1, arr2)
    return merger(tmp, arr3)


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




def findWords(board: List[List[str]], words: List[str]) -> List[str]:
    def trieBFS(node, pth, r, c, wordsPresent):
        if node.isWord:
            wordsPresent.append(pth)
            node.isWord = False

        if r < 0 or r >= rows or c < 0 or c >= cols:
            return


        tmp = board[r][c]
        node = node.children.get(tmp)
        if node is None:
            return
        board[r][c] = '#'
        trieBFS(node, pth + tmp, r + 1, c, wordsPresent)
        trieBFS(node, pth + tmp, r - 1, c, wordsPresent)
        trieBFS(node, pth + tmp, r, c + 1, wordsPresent)
        trieBFS(node, pth + tmp, r, c - 1, wordsPresent)
        board[r][c] = tmp


    trie = Trie2()
    node = trie.root
    wordsPresent = []
    for word in words:
        trie.insert(word)

    rows, cols = len(board), len(board[0])
    for r in range(rows):
        for c in range(cols):
            trieBFS(node, "", r, c, wordsPresent)
    return wordsPresent





class FileSystem:
    def __init__(self):
        self.trie = dict()

    def createPath(self, path: str, value: int) -> bool:
        t = self.trie
        path = path.split("/")[1:]
        for s in path[:-1]:
            if s not in t:
                return False
            t = t[s]

        if path[-1] in t:
            return False
        else:
            t[path[-1]] = {'_end': value}
        return True

    def get(self, path: str) -> int:
        t = self.trie
        path = path.split("/")[1:]
        for s in path:
            if s not in t:
                return -1
            t = t[s]
        return t['_end']





# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
class Codec:
    def serialize(self, root: Optional[TreeNode]):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """

        def preOrder(node):
            if node is None:
                nodes.append('#')
            else:
                nodes.append(str(node.val))
                preOrder(node.left)
                preOrder(node.right)

        nodes = []
        preOrder(root)
        return ' '.join(nodes)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """
        data = data.split(' ')

        def build(data):
            print(data[0])
            if data[0] == '#':
                data.pop(0)
                return None
            val = data[0]
            data.pop(0)
            node = TreeNode(val=val, left=build(data), right=build(data))
            return node

        return build(data)


class WordDictionary:

    def __init__(self):
        self.trie = dict()

    def addWord(self, word: str) -> None:
        trie = self.trie
        for c in word:
            if c not in trie:
                trie[c] = dict()
            trie = trie[c]
        trie['_end'] = dict()

    def search(self, word: str) -> bool:
        trie = self.trie
        for c in word:
            if c == '.':
                return any(search(self, word[1:]) for key in trie if key != '_end')
            if c not in trie:
                return False
            trie = trie[c]
        return '_end' in trie


class TrieNode():
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.isWord = False


class Trie2:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        t = self.root
        for c in word:
            t = t.children[c]
        t.isWord = True


class Trie:
    def __init__(self):
        self.trie = dict()

    def insert(self, word: str) -> None:
        t = self.trie
        for c in word:
            if c not in t:
                t[c] = {}
            t = t[c]
        t['_end'] = True

    def search(self, word: str) -> bool:
        t = self.trie
        for c in word:
            if c not in t:
                return False
            t = t[c]
        return '_end' in t

    def startsWith(self, prefix: str) -> bool:
        t = self.trie
        for c in prefix:
            if c not in t:
                return False
            t = t[c]
        return True

