def maxSubarraySumCircular(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # LC Medium: https://leetcode.com/problems/maximum-sum-circular-subarray/
    # Solution 1 for max subarray - using kadanes algorithm
    # The goal is to keep track of total sum and total min
    
    