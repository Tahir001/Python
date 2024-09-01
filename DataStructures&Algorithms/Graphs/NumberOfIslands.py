# Graphs can be given as Matrices as well
from collections import deque

def count_number_of_islands(grid):
    if not grid:
        return 0
    
    # counting the number of islands
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    visited = set()
    
    def bfs(i,j):
        queue = deque([(i,j)])
        while queue:
            r, c = queue.popleft()
            directions = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
            for nr, nc in directions:
                # if it's a land, keep going and adding it to the visited set
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1" and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
                    
    # So we iterate over each element in the matrix 
    for i in range(rows):
        for j in range(cols):
            # If we find a new land, lets explore it 
            if grid[i][j] == "1" and (i,j) not in visited:
                island_count += 1
                visited.add((i,j))
                bfs(i,j)
                
    return island_count
"""
Example: 

grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Number of Islands = 1
"""