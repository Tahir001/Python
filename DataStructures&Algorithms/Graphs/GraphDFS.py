def graph_dfs_traversal(graph, source):
    # here the graph is represented as a Adjacency List 
    stack = [source]
    result = []
    visited = set()
    while len(stack) > 0:
        
        current_node = stack.pop()

        if current_node not in visited:
            
            # Add it to the result and set
            result.append(current_node)
            visited.add(current_node)
            
            # Now lets explore this node 
            for nieghbour in graph[current_node]:
                # Get all of it's neighbours
                stack.append(nieghbour)
    return result 

# Lets try it out on an example:
graph1 = {
    'a':['c', 'b'],
    'b':['d'],
    'c':['e'],
    'd':['f'],
    'e':[],
    'f':[]
}

print(graph_dfs_traversal(graph1, 'a'))