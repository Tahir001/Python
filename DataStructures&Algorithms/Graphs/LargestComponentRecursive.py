def LargestComponentRecursive(graph):
    
    max_count = 0
    visited = set()
    for node in graph:

        if node not in visited:
            size = 0
            # Do dfs 
            size = dfs(graph, node, visited, size)
            max_count  = max(max_count, size)

    return max_count

def dfs(graph, src, visited = set(), size = 0):

    if src is None or graph is None:
        return 0 
    
    if src not in visited: 
        size += 1
        visited.add(src)

    for neighbour in graph[src]:
        if neighbour not in visited:
            size = dfs(graph, neighbour, visited, size)

    return size 
    
# Test Case 7: Complex graph
graph1 = {
    'a': ['b', 'c'],
    'b': ['a', 'd'],
    'c': ['a'],
    'd': ['b'],
    'e': ['f'],
    'f': ['e'],
    'g': []
}

print(LargestComponentRecursive(graph1))