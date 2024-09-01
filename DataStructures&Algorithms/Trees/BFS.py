from collections import deque 

def levelOrder(root):

    # If not root, just return an empty list.
    if not root:
        return []
    
    # Setup the queue 
    result = []
    queue = deque([root])

    # While there's nodes in the queue
    while queue:
        
        # Calculate the current level
        level_size = len(queue)
        current_level = []

        # Go over each node in that level 
        for _ in range(level_size):

            # Extract the first node and explore it 
            node = queue.popleft()
            current_level.append(node.val)

            # Add it's childern if they exsist 
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
        # Append the current level to the result 
        result.append(current_level)

    return result