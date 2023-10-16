def maxSubarraySumCircular(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # LC Medium: https://leetcode.com/problems/maximum-sum-circular-subarray/
    # Solution 1 for max subarray - using kadanes algorithm
    currMax, globMax = 0, nums[0]
    currMin, globMin = 0, nums[0]
    total = 0

    for n in nums:
        # Update total sum of the array
        total += n

        # Update current values 
        currMax = max(currMax + n, n)
        currMin = min(currMin + n, n)

        # Update globals 
        globMax = max(globMax, currMax)
        globMin = min(globMin, currMin)

    
    return max(globMax, (total - globMin)) if globMax > 0 else globMax



    