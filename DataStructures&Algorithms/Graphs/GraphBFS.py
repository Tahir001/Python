from collections import deque

def graph_bfs_traversal(graph, source):
    
    # Here the graph is represented as an Adjacency List
    queue = deque([source])
    result = []
    visited = set()

    while queue:
        
        current_node = queue.popleft()

        if current_node not in visited:
            # Add it to the result and set
            result.append(current_node)
            visited.add(current_node)

            # Now let's explore this node
            for neighbour in graph[current_node]:
                # Get all of its neighbours if they are not in visited set 
                if neighbour not in visited:
                    queue.append(neighbour) 

    return result 

# Let's try it out on an example:
graph1 = {
    'a': ['c', 'b'],
    'b': ['d'],
    'c': ['e'],
    'd': ['f'],
    'e': [],
    'f': []
}

print(graph_bfs_traversal(graph1, 'a'))
