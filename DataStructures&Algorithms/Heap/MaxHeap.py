class MaxHeap:
    
    def __init__(self):
        self.heap = [-1]
        
    def add(self, val):
        # Add element to the MaxHeap
        # Add to the end of the heap and then bubble up
        
        self.heap.append(val)
        # Get the index
        i = len(self.heap) - 1
        
        while i > 0 and self.heap[i//2] < self.heap[i]:
            # If parent is less than child
            # swap the nodes
        
            temp = self.heap[i//2] 
            self.heap[(i//2)] = self.heap[i]
            self.heap[i] = temp
            i = i // 2

        return self.heap[1:]
        
    def remove(self):
        # remove the priority queue guy 
        # So we are removing the top elelment
        # We replace it with the last element 
        # Then we bubble down as needed -> (if right child exsits and is bigger, go down there. )
        # If left child, and smaller, go there
        # Else: break
        
        if len(self.heap) == 1:
            return None
        if len(self.heap) == 2:
            return self.heap.pop()
        
        # Take out the first guy 
        priority_element = self.heap[1]
        
        # Replace the elemnt in the heap with last guy 
        self.heap[1] = self.heap.pop()
        
        i = 1
        # bubble down
        
        heap_boundary = len(self.heap)
        while 2*i < len(self.heap) and i > 0:
            
            # if the parent is less than it's right child -> Bubble down.
            if ((2*i + 1) < len(self.heap) and (
                self.heap[(2*i + 1)] > self.heap[(2*i)]) and (
                self.heap[i] < self.heap[(2*i + 1)])):

                # Swap the elements 
                temp = self.heap[i]
                self.heap[i] = self.heap[(2*i + 1)]
                self.heap[(2*i + 1)] = temp
                i = (2*i + 1)
            
            # Left child -> if left child is greater than parent.. swap it 
            elif self.heap[i] < self.heap[2*i]:
                temp = self.heap[i]
                self.heap[i] = self.heap[2*i]
                self.heap[2*i] = temp
                i = 2*i
            
            # else -> break, we've reached the position it needs to be 
            else:
                break
            
        return (priority_element, self.heap[1:])
        