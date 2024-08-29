from ConvertEdgeListToAdjacencyList import ConvertToAdjacencyList

def shortestPath(graph, source, dst):
    # Takes in an adjacency list graph and finds the shortest path from source to destination node
    
    # Approach: We have the source node..
    # we can go over the neighbours of the source node.. 
    # For each neighbour, we can do DFS and see if it reaches dstination, and how many edges it took to get there. 
    
    paths = []
    for neighbour in graph[source]:
        paths.append(bfs(graph, neighbour, dst))
    return min(paths)


def bfs(graph, src, dst):
    visited = set()
    size = 0

    queue = [src]
    while queue:
        curr = queue.pop(0)
        if curr == dst:
            return size 
        
        if curr not in visited:
            size += 1
            visited.add(curr)

            for neighbour in graph[curr]:
                queue.append(neighbour) 
    return size 

edges = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'C'),
    ('B', 'D'),
    ('C', 'D'),
    ('D', 'E'),
    ('C', 'E')
]

graph1 = ConvertToAdjacencyList(edges)

# Example usage of shortestPath
source = 'A'
destination = 'E'
shortest_path = shortestPath(graph1, source, destination)
print(f"Shortest path from {source} to {destination}: {shortest_path}")




