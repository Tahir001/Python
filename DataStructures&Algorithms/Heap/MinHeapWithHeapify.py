# MinHeap Class Implementation
class MinHeap:
    def __init__(self):
        self.heap = [0]  # Initialize heap with a dummy value at index 0

    def push(self, val: int) -> None:
        """Pushes a value onto the heap."""
        self.heap.append(val)
        self._bubble_up(len(self.heap) - 1)

    def pop(self) -> int:
        """Pops the smallest value off the heap."""
        if len(self.heap) <= 1:
            return -1
        if len(self.heap) == 2:
            return self.heap.pop()
        
        # Move the last element to the root and bubble it down.
        root = self.heap[1]
        self.heap[1] = self.heap.pop()
        self._bubble_down(1)
        return root

    def top(self) -> int:
        """Returns the smallest value without popping it."""
        return self.heap[1] if len(self.heap) > 1 else -1

    def heapify(self, nums: List[int]) -> None:
        """Transforms a list into a heap in-place."""
        self.heap = [0] + nums
        for i in reversed(range(1, len(self.heap) // 2 + 1)):
            self._bubble_down(i)

    def _bubble_up(self, index: int) -> None:
        parent = index // 2
        while index > 1 and self.heap[parent] > self.heap[index]:
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            index = parent
            parent = index // 2

    def _bubble_down(self, i):
        # While we have a left child
        while (2 * i) < len(self.heap) and i < len(self.heap):
            
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
        return None