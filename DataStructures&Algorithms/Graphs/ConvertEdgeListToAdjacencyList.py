def ConvertToAdjacencyList(edge_lists_graph):
    # Takes in an edge list graph representation and 
    # Converts it to an adjacency list graph representation

    hashmap = {}

    for edges in edge_lists_graph:

        node1, node2 = edges[0], edges[1]

        if node1 not in hashmap:
            hashmap[node1] = []
        hashmap[node1].append(node2)

        if node2 not in hashmap:
            hashmap[node2] = []
        hashmap[node2].append(node1)

    return hashmap

if __name__ == '__main__': 
    edges1 = [['a','b'], ['a', 'c'], ['d', 'c']]
    edges2 = [['a', 'b'], ['a', 'c'], ['a', 'd'], ['a', 'e']]
    print(ConvertToAdjacencyList(edges1))
    print(ConvertToAdjacencyList(edges2))