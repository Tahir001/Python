def HasPathRecursive(graph, src, dst, visit = set()):

    if src == dst:
        return True 
    
    if src not in visit:
        visit.add(src)
    
        for neighbour in graph[src]:
            if HasPathRecursive(graph, neighbour, dst, visit):
                return True

    return False 


# Example usage:
graph1 = {
    'a': ['c', 'b'],
    'b': ['d'],
    'c': ['e'],
    'd': ['f'],
    'e': [],
    'f': []
}

if __name__ == '__main__':
    print(HasPathRecursive(graph1, 'a', 'f'))
