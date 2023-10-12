def removeElement(nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    LeetCode 27: https://leetcode.com/problems/remove-element/description/
    """
    # Given an integer array and integer val, remove all occurences of val in nums in-place. 
    # Traverse through the array..
    # If the value is equal to val, remove it.
    # We have to do it in place, so play around with indices. 
    i=0 
    while i < len(nums):
        # Extract current value 
        curr = nums[i]
        if curr == val:
            nums.pop(i)
            i -= 1
        i += 1
    return len(nums)

            


    