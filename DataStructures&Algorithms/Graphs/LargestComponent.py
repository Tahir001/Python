def LargestGraphComponent(graph):
    # We need to know how many compoenents are there 
    # For each component, we can do some traversal algorithm (BFS or DFS) 
    # For each component, we will have to keep track of the size of that component
    # WE will need to return the largest component size 

    if graph is None:
        return 0

    visited = set()
    curr_count, max_count = 0, 0

    # Iterate over each node of the graph
    for node in graph:

        # if this is a new node -> New component
        if node not in visited: 
            # New component 
            curr_count = 0
        
            # begin exploring this node using DFS
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    curr_count += 1
                for neighbour in graph[curr]:
                    if neighbour not in visited:
                        stack.append(neighbour)
            max_count  = max(curr_count, max_count)

    return max_count

def test_LargestGraphComponent():
    # Test Case 1: Null graph
    graph = None
    assert LargestGraphComponent(graph) == 0, "Test Case 1 Failed"

    # Test Case 2: Empty graph
    graph = {}
    assert LargestGraphComponent(graph) == 0, "Test Case 2 Failed"

    # Test Case 3: Single node graph
    graph = {'a': []}
    assert LargestGraphComponent(graph) == 1, "Test Case 3 Failed"

    # Test Case 4: Disconnected nodes
    graph = {
        'a': [],
        'b': [],
        'c': []
    }
    assert LargestGraphComponent(graph) == 1, "Test Case 4 Failed"

    # Test Case 5: One connected component
    graph = {
        'a': ['b'],
        'b': ['a', 'c'],
        'c': ['b']
    }
    assert LargestGraphComponent(graph) == 3, "Test Case 5 Failed"

    # Test Case 6: Multiple connected components
    graph = {
        'a': ['b'],
        'b': ['a'],
        'c': ['d'],
        'd': ['c'],
        'e': []
    }
    assert LargestGraphComponent(graph) == 2, "Test Case 6 Failed"

    # Test Case 7: Complex graph
    graph = {
        'a': ['b', 'c'],
        'b': ['a', 'd'],
        'c': ['a'],
        'd': ['b'],
        'e': ['f'],
        'f': ['e'],
        'g': []
    }
    assert LargestGraphComponent(graph) == 4, "Test Case 7 Failed"

    print("All test cases passed!")

# Run the test cases
test_LargestGraphComponent()
