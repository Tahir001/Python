def graph_dfs_traversal(graph, source, visited = set()):
    # here the graph is represented as a Adjacency List

    if graph is None or source is None:
        return False 
    
    if source in visited:
        return 
    
    visited.add(source)
    result.append(source)

    for neighbour in graph[source]:
        # Get all of its neighbours if they are not in visited set 
        # (This is better for dealing with cycles implicitly )
        if neighbour not in visited:
            graph_dfs_traversal(graph, neighbour, visited)
    
    return None 

# Lets try it out on an example:
graph1 = {
    'a':['c', 'b'],
    'b':['d'],
    'c':['e'],
    'd':['f'],
    'e':[],
    'f':[]
}
result = []
print(graph_dfs_traversal(graph1, source= 'a'))
print(result)