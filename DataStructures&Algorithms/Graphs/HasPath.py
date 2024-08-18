def HasPath(graph, src, dst):
    # This function takes in a graph, a source node, a destination node 
    # And returns if there is a valid path from source to destination 

    # Depth First Search 
    stack = [src]
    visited = set()

    while stack:

        curr = stack.pop()
        for neighbour in graph[curr]:
            
            if neighbour == dst:
                return True
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    
    return False

# Lets try it out on an example:
graph1 = {
    'a':['c', 'b'],
    'b':['d'],
    'c':['e'],
    'd':['f'],
    'e':[],
    'f':[]
}
if __name__ == "__main__":

    print(HasPath(graph1, 'a', 'f'))