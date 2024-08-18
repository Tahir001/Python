def count_zeros(n, count=0):
    
    if n < 10:
        if n == 0:
            count += 1
        return count
    
    number = n % 10
    if number == 0:
        count += 1
        
    new_n = n//10

    return count_zeros(new_n, count)

print(count_zeros(102304005)) # returns 4 
print(count_zeros(0)) # returns 1
print(count_zeros(10023)) # returns 2