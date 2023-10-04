def maxSubArray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """

    # Kadane's Algorithm - O(N) time complexity
    # Goal is to do 1 pass
    # Essentially keeping a running total and if its less than 0, disregard it and start again
    max_sum = nums[0]
    curr_sum = 0

    for num in nums:
        # Basically set it equal to 0 if it's current running total is negative
        curr_sum = max(0, curr_sum)
        curr_sum += num
        max_sum = max(curr_sum, max_sum)

    return max_sum

    # OTHER APPROACHES BELOW. 
    # O(N**2) time complexity 
    # Brute Force
    # max_sum = nums[0]
    # for i in range(len(nums)):
    #     curr_sum = 0
    #     for j in range(i, len(nums)):
    #         curr_sum += nums[j]
    #         max_sum = max(curr_sum, max_sum)
    # return max_sum 

    # Approach Number 3 -> Also O(N) time complexity
    # Sliding window -> Suppose we want to return the indexes of the max subarray
    # L, R = 0, 0
    # currSum, maxSum = 0, nums[0]
    # maxL, maxR = 0,0

    # for number in nums:
    #     # If the current sum is less than 0, basically close the window 
    #     if currSum < 0: 
    #         currSum = 0
    #         L = R
        
    #     # If it's not less than 0... we need to adjust our pointers
    #     currSum += number
    #     if currSum > maxSum:
    #         maxSum = currSum
    #         maxL = L
    #         maxR = R
    #     # Update the R at the end, once we have finished with this number
    #     R += 1

    # return [maxL, maxR]

array = [4,-1,2,-7,3,4]
maxSubArray(array)
print(maxSubArray(array))