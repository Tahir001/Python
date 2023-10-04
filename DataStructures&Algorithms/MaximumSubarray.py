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
    

        # O(N**2) time complexity 
        # Brute Force
        # max_sum = nums[0]
        # for i in range(len(nums)):
        #     curr_sum = 0
        #     for j in range(i, len(nums)):
        #         curr_sum += nums[j]
        #         max_sum = max(curr_sum, max_sum)
        # return max_sum 