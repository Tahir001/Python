class MinHeap:
    
    def __init__(self):
        # Initialize a heap
        # We keep the first index void
        self.heap = [-1]
        
    
    def add(self, val):
        # The idea is to add it to the end, and then bubble up 
        
        # Add the elemnt to the heap, and get the starting index
        self.heap.append(val)
        i = len(self.heap) - 1 
        
        # Now we bubble up 
        while i > 1 and self.heap[i] < self.heap[(i//2)]:
            # Swap the two nodes 
            temp = self.heap[(i//2)]
            self.heap[(i//2)] = self.heap[i]
            self.heap[i] = temp
            # Update i once finished 
            i = i // 2
            
        return self.heap[1:]
        
    def remove(self):
        # The idea here is to what
        # To remove the elemnt, we see which one is the minimum value 
        # Either the right side or the left side 
        # Then we bubble down until we are in the correct place
        
        if len(self.heap) == 1:
            return None 
        if len(self.heap) == 2:
            return self.heap.pop()
        
        # Move the last value to the first value 
        self.heap[1] = self.heap.pop()
        i = 1
        
        # While we have a left child
        while (2 * i) < len(self.heap):
            
            #if the right child exsists and it's smaller than the left child
            # AND if the parent is greater than the child -> bubble down
            if (2*i + 1) < len(self.heap) and (
                self.heap[(2*i + 1)] < self.heap[(2*i)]) and (
                self.heap[i] > self.heap[(2*i +1)]):
                
                # Swap the parent with the right child
                temp = self.heap[i]
                self.heap[i] = self.heap[(2*i + 1)]
                self.heap[(2*i + 1)] = temp
                i = 2*i + 1
                
            # if the left child is smaller 
            elif self.heap[i]  > self.heap[2*i]:
                # Swap the parent with the left child
                temp = self.heap[i]
                self.heap[i] = self.heap[2*i]
                self.heap[2*i] = temp
                i = 2*i
            else:
                # We are where we want to be 
                break
                
        return self.heap[1]