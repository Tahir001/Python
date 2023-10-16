def removeDuplicatesFromSortedArray(nums):
    """
    :type nums: List[int]
    :rtype: int
    Leetcode #26: https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/
    """
    # Array is sorted...
    # Keep track of if it already exists in the array... # Hashmap ?
    # If it doesn't exists, keep it, if it does exists, remove it.
    # Need to remove the duplicates in PLACE.

    hashmap = {}
    i = 0
    while i < len(nums):
        val = nums[i]
        if val not in hashmap:
            hashmap[val] = 1
        else:
            nums.pop(i)
            i -= 1
        i += 1
    return len(nums)
