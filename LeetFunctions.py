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
import re


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


def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    curr = ListNode(0)
    head = curr

    while list1 and list2:
        if list1.val < list2.val:
            curr.next = list1
            list1 = list1.next
        else:
            curr.next = list2
            list2 = list2.next
        curr = curr.next

    if list1:
        curr.next = list1

    if list2:
        curr.next = list2

    return head.next


def maxProfit(self, prices: List[int]) -> int:
    minP, profit = float('inf'), 0
    for p in prices:
        minP = min(minP, p)
        profit = max(profit, p - minP)
    return profit


def binarySearch(self, nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] < target:
            l = mid + 1
        elif nums[mid] > target:
            r = mid - 1
        else:
            return mid
    return -1 if target != nums[l] else l


# https://leetcode.com/problems/first-bad-version/
def isBadVersion(n):
    return False

def firstBadVersion(self, n: int) -> int:
    l, r = 1, n
    while l < r:
        mid = (l + r) // 2
        if isBadVersion(mid):
            r = mid
        else:
            l = mid + 1

    return l

# https://leetcode.com/problems/longest-palindrome/description/
def longestPalindrome(self, s: str) -> int:
    longest, hasOdd = 0, False
    count = collections.Counter(s)
    for i in count.values():
        if i % 2 == 0:
            longest += i
        else:
            longest += (i - 1)
            hasOdd = True

    return longest + 1 if hasOdd else longest




# https://leetcode.com/problems/ransom-note/
def climbStairs(self, n: int) -> int:
    one, two = 1, 1
    if n == 1:
        return 1

    for i in range(n - 1):
        three = one + two
        one = two
        two = three
    return three


def isPalindrome(self, s: str) -> bool:
    s = (re.sub(r'[^a-zA-Z0-9]', '', s)).lower()
    return s == s[::-1]


def max_sub_array_of_size_k(k, arr):
    max_sum = 0
    window_sum = 0
    for i in range(len(arr)):
        window_sum += arr[i]
        if i >= k:
            window_sum -= arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum


def isBalanced(self, root: Optional[TreeNode]) -> bool:
    balanced = [True]
    def traverse(node, balanced):
        if not node:
            return 0
        l, r = traverse(node.left, balanced), traverse(node.right, balanced)
        if abs(l - r) > 1:
            balanced[0] = False
        return 1 + max(l, r)

    traverse(root)
    return balanced[0]


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
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
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
# https://leetcode.com/problems/longest-palindromic-substring/submissions/
def longestPalindrome(self, s: str) -> str:
    longest, i = "", 0
    while i < len(s):
        l, r = i - 1, i + 1
        while l >= 0 and s[l] == s[i]:
            l -= 1
        while r < len(s) and s[r] == s[i]:
            r += 1
        i = r
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        if r - l - 1 > len(longest):
            longest = s[l + 1: r]

    return longest


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


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/submissions/
def maxProfit(self, prices: List[int]) -> int:
    maxP, maxSell = 0, 0
    for i in prices[::-1]:
        maxSell = max(i, maxSell)
        maxP = max(maxP, maxSell - i)
    return maxP


# https://leetcode.com/problems/jump-game/submissions/
def canJump(nums: List[int]) -> bool:
    maxReach, n = 0, len(nums)

    for i in range(n):
        if i > maxReach:
            return False
        maxReach = max(maxReach, i + nums[i])
        if maxReach >= n - 1:
            return True


# https://leetcode.com/problems/longest-increasing-path-in-a-matrix/submissions/
def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
    def increasingDFS(r, c):
        if dp[r][c] == math.inf:
            longestDist = 0
            for d in dirs:
                r2, c2 = r + d[0], c + d[1]
                if r2 < 0 or r2 >= m or c2 < 0 or c2 >= n or matrix[r2][c2] <= matrix[r][c]:
                    continue
                longestDist = max(longestDist, increasingDFS(r2, c2))

            dp[r][c] = longestDist
        return dp[r][c] + 1

    m, n = len(matrix), len(matrix[0])
    dirs, ans = [(1, 0), (0, 1), (-1, 0), (0, -1)], 1
    dp = [[math.inf] * n for _ in range(m)]
    for r in range(m):
        for c in range(n):
            ans = max(ans, increasingDFS(r, c))
    return ans


# https://leetcode.com/problems/is-graph-bipartite/submissions/
def isBipartite(self, graph: List[List[int]]) -> bool:
    def bipartiteBFS(node, part):
        if node in valid:
            return True

        if part == 0 and node in part1:
            return False
        if part == 1 and node in part0:
            return False

        if node in part0 or node in part1:
            return True

        if part == 0:
            part0.add(node)
        else:
            part1.add(node)

        for n in graph[node]:
            if not bipartiteBFS(n, 1 - part):
                return False
        valid.add(node)
        return True

    part0, part1, valid = set(), set(), set()
    for i in range(len(graph)):
        if not bipartiteBFS(i, 0):
            return False
    return True


# https://leetcode.com/problems/median-of-two-sorted-arrays/solutions/2471/very-concise-o-log-min-m-n-iterative-solution-with-detailed-explanation/
def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    n1, n2 = len(nums1), len(nums2)
    if n1 > n2:
        return findMedianSortedArrays(nums2, nums1)

    lo, hi = 0, n1*2
    while lo <= hi:
        mid1 = (lo + hi) // 2
        mid2 = n1 + n2 - mid1

        l1 = float('-inf') if mid1 == 0 else nums1[(mid1 - 1)//2]
        l2 = float('-inf') if mid2 == 0 else nums2[(mid2 - 1)//2]
        r1 = float('inf') if mid1 == n1*2 else nums1[mid1//2]
        r2 = float('inf') if mid2 == n2*2 else nums2[mid2//2]

        if l1 > r2:
            hi = mid1 - 1
        elif l2 > r1:
            lo = mid1 + 1
        else:
            return (max(l1, l2) + min(r1, r2)) / 2
    return -1


# https://leetcode.com/problems/bus-routes/
def numBusesToDestination(routes: List[List[int]], source: int, target: int) -> int:
    mp = collections.defaultdict(set) # maps station number to bus number

    for busID, busRoute in enumerate(routes):
        for s in busRoute:
            mp[s].add(busID)

    visited = set([source])
    bfs = [(source, 0)]
    for station, dist in bfs:
        if station == target:
            return dist
        for bus in mp[station]:
            for next_station in routes[bus]:
                if next_station not in visited:
                    bfs.append((next_station, dist + 1))
                    visited.add(next_station)
            routes[bus] = []

    return -1






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


def minProductSubarray(arr):
    minProduct = arr[0]
    currProduct = arr[0]
    for i in range(len(arr)):
        currProduct = min(currProduct * arr[i], arr[i])
        minProduct = min(minProduct, currProduct)
    return minProduct


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
    currentRate = currentRate * rate
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


# https://leetcode.com/problems/clone-graph/submissions/
def cloneGraph(self, node: 'Node') -> 'Node':
    def cloner(currNode):
        if currNode.val in visited:
            return visited[currNode.val]
        clone = Node(currNode.val)
        visited[currNode.val] = clone
        for n in currNode.neighbors:
            clone.neighbors.append(cloner(n))
        return clone

    visited = {}
    return node if not node else cloner(node)


# https://leetcode.com/problems/linked-list-cycle/submissions/
def hasCycle(self, head: Optional[ListNode]) -> bool:
    slow = fast = head

    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if fast and slow == fast:
            return True
    return False


# https://github.com/BrianChen0527
def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    def isCyclic(currCourse, visiting):
        if currCourse in visited:
            return False
        if currCourse in visiting:
            return True
        visiting.add(currCourse)

        if currCourse in coursesG:
            for c in coursesG[currCourse]:
                if isCyclic(c, visiting):
                    return True
        visited.add(currCourse)
        return False

    coursesG = collections.defaultdict(list)
    for c in prerequisites:
        coursesG[c[1]].append(c[0])

    visited = set()
    for c in coursesG:
        if isCyclic(c, set()):
            return False
    return True


def longestPalindromicSubsequence(s: str):
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            if s[r - 1] == s[n - c]:
                dp[r][c] = dp[r-1][c-1] + 1
            else:
                dp[r][c] = max(dp[r-1][c], dp[r][c-1])
        print(dp)
    return dp[-1][-1]


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
        oceansBFS(i, cols - 1, canAtlantic)

    for i in range(cols):
        oceansBFS(0, i, canPacific)
        oceansBFS(rows - 1, i, canAtlantic)

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
            top, right, bottom, left = matrix[layer][p], matrix[p][n - 1 - layer], matrix[n - 1 - layer][n - 1 - p], \
                                       matrix[n - 1 - p][layer]
            print(top, " ", right, " ", bottom, " ", left)
            matrix[layer][p], matrix[p][n - 1 - layer], matrix[n - 1 - layer][n - 1 - p], matrix[n - 1 - p][
                layer] = left, top, right, bottom


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
    return pth[k - 1]


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
def minWindow2(self, s: str, t: str) -> str:
    if len(t) > len(s):
        return ""

    chars, count, minWindow = collections.defaultdict(int), len(t), None
    for c in t:
        chars[c] += 1

    l, r = 0, 0

    while r < len(s):
        c = s[r]
        if c in chars:
            if chars[c] > 0:
                count -= 1
            chars[c] -= 1

        while count == 0 and l <= r:
            if not minWindow or r - l + 1 < len(minWindow):
                minWindow = s[l:r + 1]

            if s[l] in chars:
                chars[s[l]] += 1
                if chars[s[l]] > 0:
                    count += 1
            l += 1
        r += 1
    return minWindow if minWindow else ""


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
def insertInterval(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
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


# https://leetcode.com/problems/01-matrix/
def updateMatrix(mat: List[List[int]]) -> List[List[int]]:
    rows, cols = len(mat), len(mat[0])
    dp = [[math.inf] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                dp[r][c] = 0
                continue
            left = math.inf if c - 1 < 0 else dp[r][c - 1]
            top = math.inf if r - 1 < 0 else dp[r - 1][c]
            dp[r][c] = 1 + min(left, top)
    for r in reversed(range(rows)):
        for c in reversed(range(cols)):
            right = math.inf if c + 1 >= cols else dp[r][c + 1]
            bottom = math.inf if r + 1 >= rows else dp[r + 1][c]
            dp[r][c] = min(dp[r][c], 1 + min(right, bottom))
    return dp


# https://leetcode.com/problems/combination-sum/
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    def combinationDFS(steps, path, target):
        if target < 0:
            return
        elif target == 0:
            return allPaths.append(path)
        for i in range(len(steps)):
            combinationDFS(steps[i:], path + [steps[i]], target - steps[i])

    allPaths = []
    combinationDFS(candidates, [], target)
    return allPaths


# https://leetcode.com/problems/permutations/
def permute(self, nums: List[int]) -> List[List[int]]:
    def genPerms(p):
        if len(visited) == length:
            perms.append(p)
        for n in nums:
            if n not in visited:
                visited.add(n)
                genPerms(p + [n])
                visited.remove(n)

    length, visited, perms = len(nums), set(), []
    genPerms([])
    return perms


# https://leetcode.com/problems/accounts-merge/
def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
    def accountsDFS(idx, emails):
        if idx in visited_accounts:
            return
        visited_accounts.add(idx)
        account = accounts[idx][1:]
        for e in account:
            emails.add(e)
            for neighbor in all_emails[e]:
                accountsDFS(neighbor, emails)
        return emails

    all_emails = collections.defaultdict(list)
    accountsLen = len(accounts)

    # match each email with corresponding account indices
    for i in range(accountsLen):
        for j in range(1, len(accounts[i])):
            email = accounts[i][j]
            all_emails[email].append(i)

    # Iterate through all accounts and conduct DFS
    visited_accounts, distinct_accounts, ans = set(), [], []
    for i in range(accountsLen):
        if i in visited_accounts:
            continue
        ans.append(accounts[i][:1] + sorted(accountsDFS(i, set())))
    return ans


# https://leetcode.com/problems/sort-colors/submissions/
def sortColors(nums: List[int]) -> None:
    num0s, num1s, num2s = 0, 0, 0

    for i in range(len(nums)):
        if nums[i] == 0:
            nums[num0s], nums[i] = nums[i], nums[num0s]
            num0s += 1

    for i in range(num0s, len(nums)):
        if nums[i] == 1:
            nums[num1s + num0s], nums[i] = nums[i], nums[num1s + num0s]
            num1s += 1


# https://leetcode.com/problems/partition-equal-subset-sum/
def canPartition(nums: List[int]) -> bool:
    total, bits = 0, 1
    for n in nums:
        total += n
        bits |= bits << n

    return total % 2 == 0 and (bits >> (total // 2)) & 1


# https://leetcode.com/problems/subsets/
def subsets(self, nums: List[int]) -> List[List[int]]:
    def subsetTraversal(path, remaining):
        allSubsets.append(path)
        for i in range(len(remaining)):
            subsetTraversal(path + [remaining[i]], remaining[i + 1:])

    allSubsets = []
    subsetTraversal([], nums)
    return allSubsets


# https://leetcode.com/problems/subsets/
def subsetsFast(self, nums: List[int]) -> List[List[int]]:
    totaLen = 1 << len(nums)
    allSubsets = [[] for _ in range(totaLen)]

    for i in range(totaLen):
        for j in range(len(nums)):
            if (i >> j) & 1:
                allSubsets[i].append(nums[j])
    return allSubsets


# https://leetcode.com/problems/binary-tree-right-side-view/
def rightSideView(root: Optional[TreeNode]) -> List[int]:
    q = deque()
    q.append(root)
    rightSide = []
    while len(q) > 0:
        rightSide.append(q[-1].val)
        nextLevel = []
        for node in q:
            if node.left:
                nextLevel.append(node.left)
            if node.right:
                nextLevel.append(node.right)
        q = deque(nextLevel)
    return rightSide


# https://leetcode.com/problems/letter-combinations-of-a-phone-number/
def letterCombinations(digits: str) -> List[str]:
    def genPerms(path, remaining):
        if not remaining:
            allPerms.append(path)
            return
        for c in d[remaining[0]]:
            genPerms(path + c, remaining[1:])

    if not digits:
        return []
    d = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    allPerms = []
    genPerms("", digits)
    return allPerms


# https://leetcode.com/problems/find-all-anagrams-in-a-string/
def findAnagrams(s: str, p: str) -> List[int]:
    if len(s) < len(p):
        return []

    mp, anagramIdx = collections.defaultdict(int), []
    lenS, start, end = len(s), 0, len(p)

    for c in p:
        mp[c] += 1
    for i in range(len(p)):
        if s[i] in mp:
            mp[s[i]] -= 1

    while end <= lenS:
        if all(v == 0 for v in mp.values()):
            anagramIdx.append(start)
        if s[start] in mp:
            mp[s[start]] += 1
        if end < lenS and s[end] in mp:
            mp[s[end]] -= 1
        start, end = start + 1, end + 1
    return anagramIdx


# https://leetcode.com/problems/minimum-height-trees/
def findMinHeightTrees(n: int, edges: List[List[int]]) -> List[int]:
    in_degree, adj_list = [0] * n, [[] for _ in range(n)]
    leaves = []

    for e in edges:
        adj_list[e[0]].append(e[1])
        adj_list[e[1]].append(e[0])

    for i, g in enumerate(adj_list):
        in_degree[i] = len(g)
        if len(g) == 1:
            leaves.append(i)

    while n > 2:
        new_leaves = []
        for leaf in leaves:
            for g in adj_list[leaf]:
                in_degree[g] -= 1
                if in_degree[g] == 1:
                    new_leaves.append(g)
        n -= len(leaves)
        leaves = new_leaves[:]


# https://leetcode.com/problems/task-scheduler/
def leastInterval(tasks: List[str], n: int) -> int:
    c = collections.Counter(tasks)
    v = c.values()
    maxCount = max(v)
    minTime = (maxCount - 1) * n + 1
    for t in v:
        if t == maxCount:
            minTime += 1
    return max(minTime, len(tasks))


# https://leetcode.com/problems/k-closest-points-to-origin/
def kClosest(points: List[List[int]], k: int) -> List[List[int]]:
    return heapq.nsmallest(k, points, key=lambda x: x[0]**2 + x[1]**2)

'''
# https://leetcode.com/problems/evaluate-reverse-polish-notation/
def evalRPN(tokens):
    stack = []
    for t in tokens:
        match t:
            case '+':
                stack.append(stack.pop() + stack.pop())
            case '-':
                n1, n2 = stack.pop(), stack.pop()
                stack.append(n2 - n1)
            case '*':
                stack.append(stack.pop() * stack.pop())
            case '/':
                n1, n2 = stack.pop(), stack.pop()
                stack.append(int(n2 / n1))
            case _:
                stack.append(int(t))
    return stack[0]
'''

# https://leetcode.com/problems/rotting-oranges/
def orangesRotting(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    minSteps = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rotten, fresh = set(), set()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                rotten.add((r, c))
            elif grid[r][c] == 1:
                fresh.add((r, c))

    while fresh:
        if not rotten:
            return -1
        rotten = set((r + d[0], c + d[1]) for r, c in rotten for d in dirs if (r + d[0], c + d[1]) in fresh)
        fresh -= rotten
        minSteps += 1
    return minSteps

   
# https://leetcode.com/problems/daily-temperatures/
def dailyTemperatures(temperatures: List[int]) -> List[int]:
    dp = [0] * len(temperatures)
    for i in range(len(temperatures) - 2, -1, -1):
        currTemp, ptr = temperatures[i], i + 1
        while currTemp >= temperatures[ptr]:
            if dp[ptr] == 0:
                dp[i] = 0
                break
            ptr += dp[ptr]
        dp[i] = ptr - i
    return dp


# https://leetcode.com/problems/valid-sudoku/
def isValidSudoku(board: List[List[str]]) -> bool:
    for row in board:
        s = set()
        for num in row:
            if num == ".": continue
            if num in s:
                return False
            s.add(num)

    for c in range(9):
        s = set()
        for r in range(9):
            num = board[r][c]
            if num == ".":  continue
            if num in s:
                return False
            s.add(num)

    for i in range(3):
        for j in range(3):
            s = set()
            for r in range(3):
                for c in range(3):
                    num = board[3 * i + r][3 * j + c]
                    if num == ".":  continue
                    if num in s:
                        return False
                    s.add(num)
    return True


# https://leetcode.com/problems/group-anagrams/submissions/
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    anagrams = collections.defaultdict(list)
    for s in strs:
        tmp = str(sorted(s))
        anagrams[tmp].append(s)
    return list(anagrams.values())


 # https://leetcode.com/problems/gas-station/
def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    sum_costs, lowest_fuel, lowest_fuel_point = 0, math.inf, 0

    for i in range(len(cost)):
        cost[i] = gas[i] - cost[i]
        if i - 1 >= 0:
            cost[i] += cost[i - 1]
        if cost[i] < lowest_fuel:
            lowest_fuel_point = i
            lowest_fuel = cost[i]
        sum_costs += cost[i]

    if sum_costs < 0:
        return -1

    return 0 if lowest_fuel_point + 1 == len(cost) else lowest_fuel_point

   
# https://leetcode.com/problems/subarray-sum-equals-k/
def subarraySum(nums: List[int], k: int) -> int:
    occurrences, currSum, pastSums = 0, 0, collections.defaultdict(int)
    pastSums[0] = 1
    for i in range(len(nums)):
        currSum += nums[i]
        if currSum - k in pastSums:
            occurrences += pastSums[currSum - k]
        pastSums[currSum] += 1
    return occurrences

# https://leetcode.com/problems/remove-nth-node-from-end-of-list/
def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    n1, n2 = head, head
    for i in range(n):
        n2 = n2.next

    while n2:
        n1, n2 = n1.next, n2.next
    n1.next = n1.next.next
    return head


# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    while True:
        if root.val > p.val and root.val > q.val:
            root = root.left
        elif root.val < p.val and root.val < q.val:
            root = root.right
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


# https://leetcode.com/problems/next-permutation/
def nextPermutation(nums: List[int]) -> None:
    i = len(nums) - 1
    while i > 0:
        if nums[i - 1] < nums[i]:
            idx = i
            while idx < len(nums):
                if nums[idx] <= nums[i - 1]:
                    break
                idx += 1
            nums[idx - 1], nums[i - 1] = nums[i - 1], nums[idx - 1]
            nums[i:] = sorted(nums[i:])
            break
        i -= 1
    if i == 0:
        nums = nums.sort()


# https://leetcode.com/problems/find-the-duplicate-number/
def findDuplicate(nums: List[int]) -> int:
    tortoise, hare = nums[0], nums[0]
    while True:
        tortoise, hare = nums[tortoise], nums[nums[hare]]
        if tortoise == hare: break

    hare = nums[0]
    while tortoise != hare:
        tortoise, hare = nums[tortoise], nums[hare]
    return hare
    

# https://leetcode.com/problems/course-schedule-ii/
def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    def DFS(course, visit_status):
        if visit_status[course] == 1:  # visiting (detected cycle)
            return False
        if visit_status[course] == 2:  # visited (done with this course)
            return True
        visit_status[course] = 1
        for c in prereqs[course]:
            if not DFS(c, visit_status):
                return False
        order.append(course)
        visit_status[course] = 2
        return True

    prereqs = [[] for _ in range(numCourses)]
    order, visit_status = [], [0] * numCourses

    for p in prerequisites:
        prereqs[p[0]].append(p[1])

    for c in range(numCourses):
        if not DFS(c, visit_status):
            return []
    return order

# https://leetcode.com/problems/swap-nodes-in-pairs/
def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head

    prev, curr, head = None, head, head.next
    while curr and curr.next:
        second, third = curr.next, curr.next.next
        curr.next, second.next = third, curr
        if prev:
            prev.next = second
        prev = curr
        curr = third
    return head


# https://leetcode.com/problems/path-sum-ii/
def pathSum(root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    def pathDFS(node, path, target):
        if target == 0:
            paths.append(path)
        if target < 0:
            return
        if node.left:
            pathDFS(node.left, path + [node.left.val], target - node.left.val)
        if node.right:
            pathDFS(node.right, path + [node.right.val], target - node.right.val)
    if not root:
        return []
    paths = []
    pathDFS(root, [], targetSum)


# https://leetcode.com/problems/rotate-array/submissions/
def rotate(nums: List[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    diff = len(nums) - (k % len(nums))
    print(nums[diff:] + nums[:diff])
    nums[:] = nums[diff:] + nums[:diff]


# https://leetcode.com/problems/odd-even-linked-list/
def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
        return head
    odd, even, evenHead = head, head.next, head.next
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = evenHead

    return head


# https://leetcode.com/problems/decode-string/
def decodeString(s: str) -> str:
    stack, currStr, currNum = [], '', 0

    for c in s:
        if c == '[':
            stack.append(currStr)
            stack.append(currNum)
            currStr, currNum = '', 0
        elif c == ']':
            prevNum = stack.pop()
            prevStr = stack.pop()
            currStr = prevStr + prevNum * currStr
        elif c.isnumeric():
            currNum = currNum * 10 + int(c)
        else:
            currStr += c
    return currStr


# https://leetcode.com/problems/merge-intervals/submissions/
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    idx = 1
    while True:
        if idx >= len(intervals):
            break
        while idx < len(intervals) and intervals[idx - 1][1] >= intervals[idx][0]:
            intervals[idx - 1][1] = max(intervals[idx - 1][1], intervals[idx][1])
            del intervals[idx]
        idx += 1
    return intervals


# https://leetcode.com/problems/house-robber/submissions/
def rob1(self, nums: List[int]) -> int:
    prev1, prev2, maxRob = 0, 0, 0
    for h in nums:
        maxRob = max(prev2 + h, prev1)
        prev2, prev1 = prev1, maxRob
    return maxRob


# https://leetcode.com/problems/house-robber-ii/submissions/
def rob2(self, nums: List[int]) -> int:
    def robHelper(houses):
        prev1, prev2, maxRob = 0, 0, 0
        for h in houses:
            maxRob = max(prev2 + h, prev1)
            prev2, prev1 = prev1, maxRob
        return maxRob

    if len(nums) < 3:
        return max(nums)
    return max(robHelper(nums[1:]), robHelper(nums[:-1]))


# https://leetcode.com/problems/house-robber-iii/submissions/
def rob3(self, root: Optional[TreeNode]) -> int:
    def robHelper(node):
        if not node:
            return (0,0)
        else:
            l, r = robHelper(node.left), robHelper(node.right)
            return (max(node.val + l[1] + r[1], l[0] + r[0]), l[0] + r[0])
    return max(robHelper(root))

# https://leetcode.com/problems/contiguous-array/
def findMaxLength(nums: List[int]) -> int:
    maxLen, counter, counterDic = 0, 0, collections.defaultdict(int)
    for i, n in enumerate(nums):
        if n:
            counter += 1

        else:
            counter -= 1

        if counter in counterDic:
            maxLen = max(maxLen, i - counterDic[counter])
        else:
            counterDic[counter] = i
    return maxLen


# https://leetcode.com/problems/maximum-width-of-binary-tree/
def widthOfBinaryTree(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    maxWidth, levelOrderQ = 0, deque()
    levelOrderQ.append((root, 0))
    while levelOrderQ:
        maxWidth = max(maxWidth, levelOrderQ[-1][1] - levelOrderQ[0][1])
        for i in range(len(levelOrderQ)):
            node, idx = levelOrderQ[0][0], levelOrderQ[0][1]
            if node.left:
                levelOrderQ.append((node.left, idx * 2))
            if node.right:
                levelOrderQ.append((node.right, idx * 2 + 1))
            levelOrderQ.popleft()
    return maxWidth

# https://leetcode.com/problems/find-k-closest-elements/
def findClosestElements(arr: List[int], k: int, x: int) -> List[int]:
    result, i = deque([x]), arr.index(x)
    left, right = i - 1, i + 1

    while len(result) < k:
        if left < 0:
            result.append(arr[right])
            right += 1
        elif right >= len(result):
            result.appendleft(arr[left])
            left -= 1
        else:
            if (arr[right] - x) < (x - arr[left]):
                result.append(arr[right])
                right += 1
            else:
                result.appendleft(arr[left])
                left -= 1
    return result


# https://leetcode.com/problems/asteroid-collision/
def asteroidCollision(asteroids: List[int]) -> List[int]:
    stack = []
    for asteroid in asteroids:
        if asteroid > 0:
            stack.append(asteroid)
        if not stack:
            continue
        else:
            while stack and abs(asteroid) > stack[-1]:
                stack.pop()
            if abs(asteroid) == stack[-1]:
                stack.pop()
    return stack


# https://leetcode.com/problems/reverse-integer/
def reverse(x: int) -> int:
    intMax = 2147483647
    negative = True if x < 0 else False
    x = abs(x)
    x = float(str(abs(x))[::-1])
    if negative:
        x *= -1
    if x < -(intMax + 1) or x > intMax:
        return 0
    return int(x)


# https://leetcode.com/problems/kth-largest-element-in-an-array/
def findKthLargest(nums: List[int], k: int) -> int:
    pivot = random.choice(nums)

    higher = [x for x in nums if x > pivot]
    if len(higher) >= k:
        return findKthLargest(higher, k)

    mid = [x for x in nums if x == pivot]
    if len(higher) + len(mid) >= k:
        return mid[0]

    lower = [x for x in nums if x < pivot]
    return findKthLargest(lower, k - len(higher) - len(mid))


#　https://leetcode.com/problems/maximal-square/
def maximalSquare(self, matrix: List[List[str]]) -> int:
    rows, cols, maxSize = len(matrix), len(matrix[0]), 0
    dp = [[0] * (cols + 1) for r in range(rows + 1)]
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if matrix[r - 1][c - 1] != "0":
                dp[r][c] = min(dp[r - 1][c - 1], dp[r - 1][c], dp[r][c - 1]) + 1
                maxSize = max(maxSize, dp[r][c])
    return maxSize ** 2


# https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
def zigzagLevelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    res, level, q = [], 0, deque([root])
    while q:
        currLevel = []
        if level % 2:
            for i in range(len(q) - 1, -1, -1):
                currLevel.append(q[i].val)
        else:
            currLevel = [node.val for node in q]
        res.append(currLevel)
        level += 1
        for i in range(len(q)):
            if q[0].left:
                q.append(q[0].left)
            if q[0].right:
                q.append(q[0].right)
            q.popleft()

    return res


# https://leetcode.com/problems/path-sum-iii/
def pathSum(root: Optional[TreeNode], targetSum: int) -> int:
    def pathSumDFS(node, currSum, pathSums):
        currSum += node.val
        nonlocal count
        if currSum - targetSum in pathSums:
            count += pathSums[currSum - targetSum]

        pathSums[currSum] += 1
        if node.left:
            pathSumDFS(node.left, currSum, pathSums)
        if node.right:
            pathSumDFS(node.right, currSum, pathSums)
        pathSums[currSum] -= 1

    if not root:
        return 0
    pathSums, count = collections.defaultdict(int), 0
    pathSums[0] = 1
    pathSumDFS(root, 0, pathSums)
    return count


# https://leetcode.com/problems/powx-n/
def myPow(x: float, n: int) -> float:
    if not n:
        return 1.0
    if n < 0:
        n = -n
        x = 1 / x
    num = 1
    while n:
        if n & 1:
            num *= x
        x *= x
        n >>= 1
    return num


# https://leetcode.com/problems/search-a-2d-matrix/
def searchMatrix1(matrix: List[List[int]], target: int) -> bool:
    col1 = [row[0] for row in matrix]
    targetRow = bisect.bisect_left(col1, target)
    if targetRow < len(matrix) and matrix[targetRow][0] == target:
        return True
    targetRow = max(targetRow - 1, 0)
    targetCol = bisect.bisect_left(matrix[targetRow], target)
    return targetCol < len(matrix[0]) and matrix[targetRow][targetCol] == target


# https://leetcode.com/problems/search-a-2d-matrix/
def searchMatrix2(matrix: List[List[int]], target: int) -> bool:
    rows, cols = len(matrix), len(matrix[0])
    if target < matrix[0][0] or target > matrix[rows - 1][cols - 1]:
        return False

    l, r = 0, rows * cols - 1

    while l <= r:
        mid = (l + r) // 2
        if matrix[mid // cols][mid % cols] > target:
            r = mid - 1
        elif matrix[mid // cols][mid % cols] < target:
            l = mid + 1
        else:
            return True
    return False


# https://leetcode.com/problems/largest-number/
def largestNumber(nums: List[int]) -> str:
    def compareDigits(n1, n2):
        if n1 + n2 > n2 + n1:
            return -1
        elif n1 + n2 < n2 + n1:
            return 1
        else:
            return 0

    nums = list(map(str, nums))
    ans = "".join(sorted(nums, key=functools.cmp_to_key(compareDigits)))
    return ans if ans[0] != '0' else '0'


# https://leetcode.com/problems/decode-ways/
def numDecodings(s: str) -> int:
    dp = [0] * (len(s) + 2)
    prev1, prev2, curr = 1, 1, 0

    for i in range(len(s)):
        curr = 0
        if s[i] != '0':
            curr = prev1

        if i > 0 and s[i - 1] != '0' and int(s[i - 1 : i + 1]) <= 26:
            curr += prev2

        prev2, prev1 = prev1, curr
    return curr


# https://leetcode.com/problems/reorder-list/
def reorderList(head: Optional[ListNode]) -> None:
    slow, fast = head, head

    while fast and fast.next:
        slow, fast = slow.next, fast.next.next

    newHead, curr = None, slow.next
    slow.next = None
    while curr:
        nextNode = curr.next
        curr.next = newHead
        newHead, curr = curr, nextNode

    curr = head

    while curr and newHead:
        tmp1, tmp2 = curr.next, newHead.next
        curr.next = newHead
        newHead.next = tmp1
        curr, newHead = tmp1, tmp2
    return head


# https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/
def distanceK(root: TreeNode, target: TreeNode, k: int) -> List[int]:
    def buildGraph(node):
        if node.left:
            BSTGraph[node.val].append(node.left.val)
            BSTGraph[node.left.val].append(node.val)
            buildGraph(node.left)
        if node.right:
            BSTGraph[node.val].append(node.right.val)
            BSTGraph[node.right.val].append(node.val)
            buildGraph(node.right)

    BSTGraph = collections.defaultdict(list)
    buildGraph(root)

    q = deque(BSTGraph[target.val])
    visited = {target.val}
    while k > 0:
        connections = len(q)
        for n in range(connections):
            node = q.popleft()
            for newNode in BSTGraph[node]:
                if newNode not in visited:
                    q.append(newNode)
                    visited.add(newNode)
        k -= 1
    return q


# https://leetcode.com/problems/3sum-closest/
def threeSumClosest(self, nums: List[int], target: int) -> int:
    n, minDiff, ans = len(nums), math.inf, 0
    nums = sorted(nums)
    for i in range(n - 2):
        l, r = i + 1, n - 1
        while l < r:
            currSum = nums[i] + nums[l] + nums[r]
            if abs(currSum - target) < minDiff:
                minDiff = abs(currSum - target)
                ans = currSum

            if currSum > target:
                r -= 1
            elif currSum < target:
                l += 1
            else:
                return target
    return ans


# https://leetcode.com/problems/rotate-list/
def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not k or not head or not head.next:
        return head
    listLength, curr, newHead, tmp = 0, head, head, None

    while curr:
        curr = curr.next
        listLength += 1

    rotateLength = k % listLength
    if not rotateLength:
        return head

    for i in range(listLength - rotateLength - 1):
        newHead = newHead.next

    tmp, newHead = newHead, newHead.next
    tmp.next, tmp = None, newHead

    while tmp.next:
        tmp = tmp.next
    tmp.next = head
    return newHead


# https://leetcode.com/problems/maximum-profit-in-job-scheduling/submissions/
def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    def jobsDFS(t):
        if t == n:
            return 0
        if t in dp:
            return dp[t]
        ans = jobsDFS(t + 1)

        t2 = bisect.bisect_left(startTimes, jobs[t][1])
        ans = max(ans, jobs[t][2] + jobsDFS(t2))
        dp[t] = ans
        return ans

    n, maxTime, maxP, dp = len(startTime), 0, 0, {}
    jobs = sorted(list(zip(startTime, endTime, profit)))
    startTimes = [j[0] for j in jobs]

    return jobsDFS(0)


# https://leetcode.com/problems/basic-calculator-ii/
def calculate(self, s: str) -> int:
    currNum, stack = 0, []
    sign = '+'

    for n in (s + '+'):
        if n.isspace():
            continue
        if n.isnumeric():
            currNum = currNum * 10 + int(n)
            continue

        if sign == '+':
            stack.append(currNum)
        elif sign == '-':
            stack.append(-currNum)
        elif sign == '*':
            res = stack.pop() * currNum
            stack.append(res)
        else:
            res = int(stack.pop() / currNum)
            stack.append(res)

        currNum = 0
        sign = n
    return sum(stack)


def knapsackClassic(weights, values, capacity):
    n = len(weights)
    dp = [[0] * capacity for _ in range(2)]

    for i in range(n):
        for c in range(capacity):
            if c == 0:
                continue
            elif c >= weights[i]:
                dp[(i + 1) % 2][c] = max(dp[i % 2][c], dp[(i + 1) % 2][c - weights[i]] + values[i])
            else:
                dp[(i + 1) % 2][c] = dp[i % 2][c]

    return dp[-1][-1]


# https://leetcode.com/problems/cheapest-flights-within-k-stops/
def findCheapestPrice(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    flightsG = [collections.defaultdict(int) for _ in range(n)]

    for f in flights:
        flightsG[f[0]][f[1]] = f[2]
    pq, stopsLeftWhenVisited = [(0, src, k + 1)], [0] * n  # pq = [(currPrice, currCity, stopsLeft)]

    while pq:
        currPrice, currCity, stopsLeft = heapq.heappop(pq)
        if currCity == dst:
            return currPrice
        if stopsLeftWhenVisited[currCity] >= stopsLeft:
            continue
        stopsLeftWhenVisited[currCity] = stopsLeft
        for city in flightsG[currCity]:
            heapq.heappush(pq, (currPrice + flightsG[currCity][city], city, stopsLeft - 1))
    return -1


# https://leetcode.com/problems/add-two-numbers/
def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    ans = ListNode()
    head, carry = ans, 0
    while l1 or l2:
        newVal = 0
        if not l1:
            newVal = l2.val + carry
            l2 = l2.next
        elif not l2:
            newVal = l1.val + carry
            l1 = l1.next
        else:
            newVal = l1.val + l2.val + carry
            l1, l2 = l1.next, l2.next

        ans.next = ListNode(newVal % 10)
        carry = newVal // 10
        ans = ans.next
    return head.next


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


# https://leetcode.com/problems/word-ladder/submissions/
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    alph = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    q, s, minChanges, visited = deque([(beginWord, 1)]), set(wordList), math.inf, set([beginWord])

    if endWord not in s:
        return 0

    while q:
        currWord, currSteps = q.popleft()

        for i in range(len(currWord)):
            for c in alph:
                neWord = currWord[:i] + c + currWord[i + 1:]
                if neWord in visited:
                    continue
                if neWord in s:
                    if neWord == endWord:
                        return currSteps + 1
                    q.append((neWord, currSteps + 1))
                    visited.add(neWord)
    return 0

'''
# https://leetcode.com/problems/basic-calculator/submissions/
def calculate(self, s: str) -> int:
    stack, currNum, isNeg = [], 0, False
    currSum = 0
    for c in (s + "+"):
        match c:
            case '(':
                stack.append(currSum)
                stack.append(isNeg)
                currSum, currNum = 0, 0
                isNeg = False
            case ' ':
                continue
            case ')':
                if isNeg:
                    currSum -= currNum
                    isNeg = False
                else:
                    currSum += currNum
                currNum = 0
                prevNeg = stack.pop()
                prevSum = stack.pop()
                if prevNeg:
                    currSum = prevSum - currSum
                else:
                    currSum = prevSum + currSum
                currNum = 0
            case '+':
                if isNeg:
                    currSum -= currNum
                    isNeg = False
                else:
                    currSum += currNum
                currNum = 0
            case '-':
                if isNeg:
                    currSum -= currNum
                else:
                    currSum += currNum
                isNeg = True
                currNum = 0
            case _:
                currNum = currNum * 10 + int(c)
    return currSum
'''


# https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/submissions/
def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
    def cityDijkstra(city):
        q = []
        heapq.heappush(q, (-distanceThreshold, city))
        visited, canVisit = set(), 0

        while q:
            dist, city = heapq.heappop(q)
            if city in visited:
                continue
            for dest in G[city]:
                if dest[0] not in visited and dist + dest[1] <= 0:
                    heapq.heappush(q, (dist + dest[1], dest[0]))
            canVisit += 1
            visited.add(city)

        return canVisit

    G, minReach, ans = [[] for _ in range(n)], math.inf, 0
    for e in edges:
        G[e[0]].append((e[1], e[2]))
        G[e[1]].append((e[0], e[2]))
    for i in range(n):
        r = cityDijkstra(i)
        if r <= minReach:
            minReach = r
            ans = i
    return ans


# https://leetcode.com/problems/zigzag-conversion/submissions/
def convert(self, s: str, numRows: int) -> str:
    if numRows == 1:
        return s

    ans, sLen, cycle = "", len(s), 2 * numRows - 2
    for i in range(numRows):
        c = i
        if i == 0 or i == numRows - 1:
            while c < sLen:
                ans += s[c]
                c += cycle
        else:
            while c < sLen:
                ans += s[c]
                c += cycle
                if c - 2 * i < sLen:
                    ans += s[c - 2 * i]
    return ans


# https://leetcode.com/problems/sum-of-two-integers/submissions/
def getSum(self, a: int, b: int) -> int:
    return sum([a,b])


# https://leetcode.com/problems/4sum/
def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
    def NSum(s, N, prevNums, target):
        ans = []

        if N == 2:
            l, r = s, lenN - 1
            while l < r:
                currS = nums[l] + nums[r]
                if currS > target:
                    r -= 1
                elif currS < target:
                    l += 1
                else:
                    ans += [prevNums + [nums[l], nums[r]]]
                    while l + 1 < r and nums[l] == nums[l + 1]:
                        l += 1
                    while r - 1 > l and nums[r] == nums[r - 1]:
                        r -= 1
                    l, r = l + 1, r - 1
        else:
            while s < lenN - N + 1:
                if target < sum(nums[s:s + N]):
                    break
                ans += NSum(s + 1, N - 1, prevNums + [nums[s]], target - nums[s])
                s += 1
                while s < lenN and nums[s] == nums[s - 1]:
                    s += 1
        return ans

    lenN, nums = len(nums), sorted(nums)
    return NSum(0, 4, [], target)


# https://leetcode.com/problems/integer-to-roman/submissions/
def intToRoman(self, num: int) -> str:
    d0 = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    d1 = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    d2 = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
    d3 = ["", "M", "MM", "MMM"]

    return d3[num // 1000] + d2[(num % 1000) // 100] + d1[(num % 100) // 10] + d0[num % 10]

# https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/submissions/
def strStr(self, haystack: str, needle: str) -> int:
    h, n = len(haystack), len(needle)

    for i in range(h - n + 1):

        pos = i
        while haystack[pos] == needle[pos - i]:
            pos += 1
            if pos - i == n:
                return i
    return -1


# https://leetcode.com/problems/generate-parentheses/
def generateParenthesis(n: int) -> List[str]:
    def parenthesisGenerator(k, permutation):
        if k == 0:
            permutation += ')' * len(stack)
            res.append(permutation)
            return
        if stack:
            stack.pop()
            parenthesisGenerator(k, permutation + ')')
            stack.append('(')
        stack.append('(')
        parenthesisGenerator(k - 1, permutation + '(')
        stack.pop()

    res, stack = [], []
    parenthesisGenerator(n, "")
    return res


# https://leetcode.com/problems/random-pick-with-weight/
class weightedPick:

    def __init__(self, w: List[int]):
        self.weights = itertools.accumulate(w)

    def pickIndex(self) -> int:
        randIdx = random.random()*self.weights[-1]
        return bisect.bisect_left(self.weights, randIdx)


# https://leetcode.com/problems/sort-list/
class MergeSort():
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        return self.customMergeSort(head)

    def mergeLists(self, l1: Optional[ListNode], l2: Optional[ListNode]):
        head = ListNode(0)
        currNode = head
        while l1 and l2:
            if l1.val < l2.val:
                currNode.next = l1
                currNode, l1 = currNode.next, l1.next
            else:
                currNode.next = l2
                currNode, l2 = currNode.next, l2.next
        if l1:
            currNode.next = l1
        elif l2:
            currNode.next = l2
        return head.next

    def customMergeSort(self, head):
        if not head or not head.next:
            return head

        slow, fast = head, head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        tmp = slow.next
        slow.next = None
        l1, l2 = self.customMergeSort(head), self.customMergeSort(tmp)
        return self.mergeLists(l1, l2)


# https://leetcode.com/problems/lru-cache/
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class TimeMap:

    def __init__(self):
        self.times = [collections.defaultdict(str)]

    def set(self, key: str, value: str, timestamp: int) -> None:
        currLen = len(self.times)
        if currLen < timestamp:
            self.times += [None] * (timestamp - currLen)
        self.times[timestamp - 1] = collections.defaultdict(str)
        self.times[timestamp - 1][key] = value

    def get(self, key: str, timestamp: int) -> str:
        for i in range(timestamp - 1, -1, -1):
            if i < len(self.times) and self.times[i] is not None:
                if key in self.times[i]:
                    return self.times[i][key]
        return ""


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


# https://leetcode.com/problems/maximum-frequency-stack/submissions/
class FreqStack:

    def __init__(self):
        self.numToFreq = collections.Counter()
        self.freqDict = collections.defaultdict(list)
        self.maxFreq = 0

    def push(self, val: int) -> None:
        numToFreq, freqDict = self.numToFreq, self.freqDict
        numToFreq[val] += 1
        freqDict[numToFreq[val]].append(val)
        self.maxFreq = max(self.maxFreq, numToFreq[val])

    def pop(self) -> int:
        numToFreq, freqDict = self.numToFreq, self.freqDict
        val = freqDict[self.maxFreq].pop()
        numToFreq[val] -= 1
        if not freqDict[self.maxFreq]:
            self.maxFreq -= 1

        return val


# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return "_,"

        return str(root.val) + "," + self.serialize(root.left) + self.serialize(root.right)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        def preorderTraversal():
            nonlocal data
            pos = data.find(',')
            val = data[:pos]
            data = data[pos + 1:]
            if val == '_':
                return None
            else:
                node = TreeNode(int(val))
                node.left = preorderTraversal()
                node.right = preorderTraversal()
                return node

        return preorderTraversal()


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
            return (self.large[0] - self.small[0]) / 2
        else:
            return -float(self.small[0])


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


def sql_to_java(filename):
    f = open(filename, "r")
    Lines = f.readlines()
    for line in Lines:
        if "--" in line:
            continue
        if not line.strip():
            print()
            print()
            continue
        print("\"" + line.rstrip().replace(";", "") + " \" + ")


