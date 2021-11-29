# functions file

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


"""Given an array of positive numbers and a positive number ‘k,’ 
find the maximum sum of any contiguous subarray of size ‘k’."""


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


#Given a string, find the length of the longest substring in it with no more than K distinct characters.
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

#Given a list of intervals, merge all the overlapping intervals to produce a list that has only mutually exclusive intervals.
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

#reverse a string str
def reverse_string(str):
    reversed_str = ""
    for i in range(len(str)):
        reversed_str += str[len(str)-1-i]
    return reversed_str

#Merge nums1 and nums2 into a single array sorted in non-decreasing order BY ALTERING THE FIRST GIVEN ARRAY
# https://leetcode.com/problems/merge-sorted-array/
def merge_sorted_arrays(nums1, m, nums2, n):
    new_arr = [None]*(m+n)
    c1, c2 = 0, 0
    for i in range(m+n):
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

#Given an int array nums and a target integer, return indices of the two numbers such that they add up to target.
def twoSum(self, nums, target):
    hashmap = {}
    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[nums[i]] = i

def firstRecurringNumber(arr):
    hashmap = {}
    for i in range(len(arr)):
        if(hashmap.get(arr[i])):
            return arr[i]
        hashmap[arr[i]] = 1
    return -1

