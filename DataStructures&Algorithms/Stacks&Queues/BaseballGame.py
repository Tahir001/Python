class Solution:
    def calPoints(self, operations: List[str]) -> int:
        # LC 682: https://leetcode.com/problems/baseball-game/description/
        stack=[]
        for i in operations:
            if i =='D':
                stack.append(2*stack[-1])
            elif i=='C':
                stack.pop()
            elif i=='+':
                stack.append(stack[-1]+stack[-2])
            else:
                stack.append(int(i))
        return sum(stack)
    

# Other submission
'''
# Same method, just a bit unoptimized 

class Solution(object):
    def calPoints(self, operations):
        """
        :type operations: List[str]
        :rtype: int
        """

        if len(operations) <= 1:
            return operations 

        my_stack = []
        for elm in operations: 
            print(my_stack)
            try:
                elm = int(elm)
            except:
                pass

            # Adding an element 
            if type(elm) == int:
                my_stack.append(elm)

            # Record a new score that is sum of previous two 
            if elm == '+':
                if len(my_stack) > 1:
                    new_val = my_stack[-1] + my_stack[-2]
                    my_stack.append(new_val)

            #Record a new score: multiply by last two elements and add it 
            if elm == 'D' and len(operations)>0:
                if len(my_stack) > 0:
                    new_val = my_stack[-1] * 2
                    my_stack.append(new_val)

            # Remove last element 
            if elm == 'C':
                if len(my_stack)>0:
                    my_stack.pop()
        return sum(my_stack)

'''