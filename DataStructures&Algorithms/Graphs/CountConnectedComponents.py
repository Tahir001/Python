def CountConnectedComponents(graph):

    # If the graph is null, return -1
    if graph is None:
        return -1
    
    # Define an empty stack, an empty set and a variable to keep track of count
    stack = []
    visited = set()
    count  = 0

    # We iterate over each node in the graph
    for node in graph:

        if node not in visited:
            visited.add(node)
            count += 1

            # Do DFS on this node 
            stack = [node]
            while stack:
                curr = stack.pop()

                if curr not in visited:
                    visited.add(curr)

                for neighbour in graph[curr]:
                    if neighbour not in visited:
                        stack.append(neighbour)
                        
    return count


def test_CountConnectedComponents():
    # Test Case 1: Null graph
    graph = None
    assert CountConnectedComponents(graph) == -1, "Test Case 1 Failed"

    # Test Case 2: Empty graph
    graph = {}
    assert CountConnectedComponents(graph) == 0, "Test Case 2 Failed"

    # Test Case 3: Single node graph
    graph = {'a': []}
    assert CountConnectedComponents(graph) == 1, "Test Case 3 Failed"

    # Test Case 4: Disconnected nodes
    graph = {
        'a': [],
        'b': [],
        'c': []
    }
    assert CountConnectedComponents(graph) == 3, "Test Case 4 Failed"

    # Test Case 5: One connected component
    graph = {
        'a': ['b'],
        'b': ['a', 'c'],
        'c': ['b']
    }
    assert CountConnectedComponents(graph) == 1, "Test Case 5 Failed"

    # Test Case 6: Multiple connected components
    graph = {
        'a': ['b'],
        'b': ['a'],
        'c': ['d'],
        'd': ['c'],
        'e': []
    }
    assert CountConnectedComponents(graph) == 3, "Test Case 6 Failed"

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
    assert CountConnectedComponents(graph) == 3, "Test Case 7 Failed"

    print("All test cases passed!")

# Run the test cases
test_CountConnectedComponents()