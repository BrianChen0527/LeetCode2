# functions file
import bisect
import math
import random
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


# https://leetcode.com/problems/longest-increasing-subsequence/
def lengthOfLIS(self, nums: List[int]) -> int:
    dp = []
    for num in nums:
        dp[bisect.bisect_left(dp, num, 0, len(dp))] = num
    return len(dp)


# https://leetcode.com/problems/longest-common-subsequence/
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    n1, n2 = len(text1), len(text2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]




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


# https://leetcode.com/problems/top-k-frequent-elements/
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    d = collections.defaultdict(int)

    for n in nums:
        d[n] += 1

    ans, arr = [], []
    for key in d:
        arr.append((-d[key], key))
    arr = heapq.nsmallest(k, arr, key=lambda x: x[0])
    for i in range(k):
        ans.append(arr[i][1])
    return ans


# https://leetcode.com/problems/palindromic-substrings/
def countSubstrings(self, s: str) -> int:
    totalPalindromes = 0
    length = len(s)
    for i in range(length):
        ptr1, ptr2 = i, i
        c, count = s[ptr1], 0
        while ptr2 + 1 < length and s[ptr2 + 1] == s[ptr1]:
            ptr2 += 1
            count += 1
        while ptr2 + 1 < length and ptr1 - 1 >= 0 and s[ptr2 + 1] == s[ptr1 - 1]:
            ptr2 += 1
            ptr1 -= 1
            count += 1
        totalPalindromes += count
    return totalPalindromes


# https://leetcode.com/problems/clone-graph/
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


# https://leetcode.com/problems/clone-graph/
def cloneGraph(self, node: 'Node') -> 'Node':
    def cloner(node, visited):
        if node.val in visited:
            return visited[node.val]

        newNode = Node(node.val)
        visited[node.val] = newNode

        for neighbor in node.neighbors:
            newNode.neighbors.append(cloner(neighbor, visited))
        return newNode

    if node is None:
        return None
    visited = {}
    return cloner(node, visited)


# https://github.com/BrianChen0527
def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    def isCyclic(mp, course, visiting):
        if course in nonCyclic:  # if we already know this course is not part of a cycle
            return False
        if course in visiting:  # if we've visited this course already (cycle!)
            return True
        visiting.add(course)
        for p in mp[course]:
            if isCyclic(mp, p, visiting):
                return True
        nonCyclic.add(course)
        return False

    mp = [[] for i in range(numCourses)]
    nonCyclic = set()  # nodes added to nonCyclic are non-cyclic, so if a course's prereq is a node in nonCyclic,
                       # we know this course is non-cyclic too and thus can optimize for speed
    for p in prerequisites:
        mp[p[0]].append(p[1])

    for n in range(len(mp)):
        visiting = set()
        if isCyclic(mp, n, visiting):
            return False
    return True


# https://leetcode.com/problems/pacific-atlantic-water-flow/
def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    def oceansBFS(r, c, canReach):
        canReach[r][c] = True
        for d in dirs:
            r2, c2 = r + d[0], c + d[1]
            if r2 < 0 or r2 >= rows or c2 < 0 or c2 >= cols or heights[r][c] < heights[r2][c2] or canReach[r2][c2]:
                continue
            oceansBFS(r2, c2, canReach)

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    canPacific = [[False] * len(heights[0]) for _ in range(len(heights))]
    canAtlantic = [[False] * len(heights[0]) for _ in range(len(heights))]
    rows, cols = len(heights), len(heights[0])

    for i in range(rows):
        oceansBFS(i, 0, canPacific)
        oceansBFS(i, cols-1, canAtlantic)

    for i in range(cols):
        oceansBFS(0, i, canPacific)
        oceansBFS(rows-1, i, canAtlantic)

    canReachBoth = []
    for r in range(rows):
        for c in range(cols):
            if canPacific[r][c] and canAtlantic[r][c]:
                canReachBoth.append([r, c])
    return canReachBoth


# https://leetcode.com/problems/number-of-islands/
def numIslands(self, grid: List[List[str]]) -> int:
    def islandSearch(r, c):
        if grid[r][c] == '_':
            return
        grid[r][c] = '_'

        for d in dirs:
            r2, c2 = r + d[0], c + d[1]
            if r2 < 0 or r2 >= rows or c2 < 0 or c2 >= cols or grid[r2][c2] == '0':
                continue
            islandSearch(r2, c2)

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rows, cols = len(grid), len(grid[0])
    num_islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                num_islands += 1
                islandSearch(r, c)
    return num_islands


# https://leetcode.com/problems/longest-consecutive-sequence/
def longestConsecutive(nums: List[int]) -> int:
    def dictDFS(num):
        if num in d:  # dynamic programming
            return d[num]
        if num in presentNums:
            d[num] = dictDFS(num + 1) + 1
            return d[num]
        return 0
    if not nums:
        return 0

    presentNums = set(nums)
    d = dict()

    maxConsecutive = 1
    for num in presentNums:
        maxConsecutive = max(maxConsecutive, dictDFS(num))
    return maxConsecutive


# https://leetcode.com/problems/set-matrix-zeroes/
def setZeroes(self, matrix: List[List[int]]) -> None:
    """
    We go through matrix, and if we find a 0 at (r,c), we set matrix[r,0] and matrix[0,c] to 0s
    As a result of this logic, we also need to check if there are 0s in row 0 and col 0, that way we know if the
    first row and first column need to be all zeroes as well
    ######0###
    #XXXXXXXXX
    0XXXXX0XXX
    #XXXXXXXXX
    """
    row0, col0 = False, False
    rows, cols = len(matrix), len(matrix[0])

    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                if c == 0: col0 = True
                if r == 0: row0 = True
                matrix[0][c] = 0
                matrix[r][0] = 0

    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[0][c] == 0 or matrix[r][0] == 0:
                matrix[r][c] = 0
    if col0:
        for r in range(rows):
            matrix[r][0] = 0
    if row0:
        matrix[0] = [0] * len(matrix[0])


# https://leetcode.com/problems/spiral-matrix/
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # order of directions: right -> down -> left -> up
    rows, cols = len(matrix), len(matrix[0])
    spiralArr = []
    r, c, order = 0, 0, 0
    while len(spiralArr) < rows * cols:
        print(r, " ", c)
        spiralArr.append(matrix[r][c])
        matrix[r][c] = math.inf
        r2, c2 = r + dirs[order][0], c + dirs[order][1]
        if r2 < 0 or r2 >= rows or c2 < 0 or c2 >= cols or matrix[r2][c2] == math.inf:
            order = (order + 1) % 4
        r, c = r + dirs[order][0], c + dirs[order][1]

    return spiralArr


# https://leetcode.com/problems/rotate-image/
def rotate(self, matrix: List[List[int]]) -> None:
    n = len(matrix)
    for layer in range((n + 1) // 2):
        length = n - 2 * layer
        for p in range(layer, n - 1 - layer):
            top, right, bottom, left = matrix[layer][p], matrix[p][n - 1 - layer], matrix[n - 1 - layer][n - 1 - p], matrix[n - 1 - p][layer]
            print(top, " ", right, " ", bottom, " ", left)
            matrix[layer][p], matrix[p][n - 1 - layer], matrix[n - 1 - layer][n - 1 - p], matrix[n - 1 - p][layer] = left, top, right , bottom


# https://leetcode.com/problems/word-search/
def exist(self, board: List[List[str]], word: str) -> bool:
    def wordSearch(word, r, c):
        if word == "":
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[0]:
            return False
        tmp, board[r][c] = board[r][c], '_'
        res = any(wordSearch(word[1:], r + d[0], c + d[1]) for d in dirs)
        board[r][c] = tmp
        return res


    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rows, cols = len(board), len(board[0])
    for r in range(rows):
        for c in range(cols):
            if wordSearch(word, r, c):
                return True
    return False


# https://leetcode.com/problems/coin-change/submissions/
def coinChange(self, coins: List[int], amount: int) -> int:
    dp = [math.inf] * amount
    dp[0] = 1

    for i in range(amount):
        for c in coins:
            if i - c >= 0 and dp[i - c] != math.inf:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[-1] if dp[-1] != math.inf else -1


# https://leetcode.com/problems/decode-ways/submissions/
def numDecodings(self, s: str) -> int:
    prev2, prev1 = 1, 1

    for i in range(len(s)):
        tmp = 0
        if s[i] != '0':
            tmp += prev1
        if i - 1 >= 0 and s[i - 1] != '0' and int(s[i - 1: i + 1]) <= 26:
            tmp += prev2
        prev2 = prev1
        prev1 = tmp
    return prev1


# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def lengthOfLongestSubstring(s: str) -> int:
    d = dict()
    left, right = 0, 0
    longest = 0
    while right < len(s):
        if s[right] not in d:
            d[s[right]] = right
        else:
            left = max(left, d[s[right]] + 1)
            d[s[right]] = right
        print(left, " ", right, " ", longest)
        longest = max(longest, right - left + 1)
        right += 1

    return longest


# https://leetcode.com/problems/kth-smallest-element-in-a-bst/
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    def inorderTraversal(node, pth):
        if len(pth) == k:
            return
        if node:
            inorderTraversal(node.left, pth)
            pth.append(node.val)
            inorderTraversal(node.right, pth)
    pth = []
    inorderTraversal(root, pth)
    return pth[k-1]


# https://leetcode.com/problems/longest-repeating-character-replacement/
def characterReplacement(self, s: str, k: int) -> int:
    chars = collections.defaultdict(int)
    start, maxCharCount, maxLen = 0, 0, 0
    for end in range(len(s)):
        chars[s[end]] += 1
        maxCharCount = max(maxCharCount, chars[s[end]])
        while end - start + 1 - maxCharCount > k:
            chars[s[start]] -= 1
            start += 1
        maxLen = max(maxLen, end - start + 1)
    return maxLen


# https://leetcode.com/problems/non-overlapping-intervals/
def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x: x[1])
    end, removed = -math.inf, 0
    for s, e in intervals:
        if s < end:
            removed += 1
        else:
            end = e
    return removed


# https://leetcode.com/problems/minimum-window-substring/
def minWindow(s: str, t: str) -> str:
    needLen, needs = len(t), collections.Counter(t)
    start, minSubstr = 0, ""

    for i, c in enumerate(s):
        if c in needs:
            print(needs)
            if needs[c] > 0:
                needLen -= 1
            needs[c] -= 1
        while needLen <= 0 and start < i:
            if minSubstr == "" or len(minSubstr) > i - start + 1:
                minSubstr = s[start: i + 1]
            needs[s[start]] += 1
            start += 1
            if needs[s[start]] > 0:
                needLen += 1
    return minSubstr


# https://leetcode.com/problems/diameter-of-binary-tree/
def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    def depthFinder(node):
        if node is None:
            return 0
        l, r = depthFinder(node.left), depthFinder(node.right)
        nonlocal diameter
        diameter = max(diameter, l + r)
        return 1 + max(l, r)

    diameter = 0
    depthFinder(root)
    return diameter


# https://leetcode.com/problems/insert-interval/
def insertInterval(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    def merge(i):
        while len(intervals) - 1 > i and intervals[i][1] >= intervals[i + 1][0]:
            intervals[i][1] = max(intervals[i][1], intervals[i + 1][1])
            intervals.pop(i + 1)

    if not intervals:
        return [newInterval]

    idx = 0
    while idx < len(intervals) and intervals[idx][0] < newInterval[0]:
        idx += 1
    if idx > 0 and intervals[idx - 1][1] >= newInterval[0]:
        intervals[idx - 1][1] = max(intervals[idx - 1][1], newInterval[1])
        merge(idx - 1)
    else:
        intervals.insert(idx, newInterval)
        merge(idx)
    return intervals



# https://leetcode.com/problems/implement-queue-using-stacks/        
class MyQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
        self.stack2.append(x)

    def pop(self) -> int:
        if len(self.stack1) > 0:
            return self.stack1.pop(-1)
        
        while len(self.stack2) > 0:
            self.stack1.append(self.stack2.pop(-1))
        return self.stack1.pop(-1)            
        
    def peek(self) -> int:
        if len(self.stack1) > 0:
            return self.stack1[-1]

        while len(self.stack2) > 0:
            self.stack1.append(self.stack2.pop(-1))
        return self.stack1[-1]

    def empty(self) -> bool:
        return len(self.stack2) == 0 and len(self.stack1) == 0


# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root.val > p.val and root.val > q.val:
        return self.lowestCommonAncestor(self, root.left, p, q)
    elif root.val < p.val and root.val < q.val:
        return self.lowestCommonAncestor(self, root.right, p, q)
    else:
        return root


# https://leetcode.com/problems/combination-sum-iv/
def combinationSum4(self, nums: List[int], target: int) -> int:
    dp = [0 * target + 1]
    dp[0] = 1

    for i in range(target):
        for n in nums:
            if i - n >= 0:
                dp[i] += dp[i - n]
    return dp[-1]


# https://leetcode.com/problems/basic-calculator-ii/
def calculate(self, s: str) -> int:
    pass








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




# https://leetcode.com/problems/insert-delete-getrandom-o1/
class RandomizedSet:

    def __init__(self):
        self.s = set()

    def insert(self, val: int) -> bool:
        if val in self.s:
            return False
        self.s.add(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.s:
            return False
        self.s.remove(val)
        return True

    def getRandom(self) -> int:
        return list(self.s)[random.random(0, len(self.s))]


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


class MedianFinder:

    def __init__(self):
        self.small = []  # heap for smaller half of numbers
        self.large = []  # heap for larger half of numbers

    def addNum(self, num: int) -> None:
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappushpop(self.large, num))
        else:
            heapq.heappush(self.large, -heapq.heappushpop(self.small, -num))


    def findMedian(self) -> float:
        if len(self.small) == len(self.large):
            return (self.large[0] - self.small[0])/2
        else:
            return -float(self.small[0])

