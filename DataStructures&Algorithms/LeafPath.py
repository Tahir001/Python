# Question: Determine if a path exsists from the root of the tree to a leaf node
# And then return that path 
# It may not contain any zeroes
# Each tree can only have atmost 1 path

class Node:
    
    def __init__(self, value=None, left=None, right=None):
        self.val = value
        self.left = left
        self.right = right

class TreeMaze:
    def __init__(self):
        pass 

    def LeafPath(self, root):
        # Return the path if it exsists without zeroes from the root of the tree to a leaf node

        if root is None:
            return None

        stack = [(root, [root.val])]


