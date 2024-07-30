# backtracking 
# Question: Determine if a path exsists from the root of the tree to a leaf node
# It may not contain any zeroes
# Each tree can only have atmost 1 path

class Node:
    
    def __init__(self, val=value, left=None, right=None):
        self.val = value
        self.left = left
        self.right = right

class TreeMaze:
    def path_exsists(self, root):
        # Return true if a path exsists without zeroes from the root of the tree to a leaf node
        if not root or root.val == 0:
            return False
        
        if not root.left or root.right:
            return True
        if self.path_exsists(root.left):
            return True
        if self.path_exsists(root.right):
            return True 

        return False 

