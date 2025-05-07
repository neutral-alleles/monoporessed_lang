# Sum numbers 0 to 9 and count loop executions
declare int32 sum_val
set sum_val 0
declare int32 count_executions # Variable to increment
set count_executions 0
declare int32 i # Loop variable

for i in range 10
    set sum_val add sum_val i # Actual summation
    inc count_executions      # Count how many times the loop ran
    print_int sum_val         # Use print_int for intermediate sum

# Example end state: sum_val should be 45, count_executions should be 10 

# Print the final results
print_int sum_val           # Use print_int
print_int count_executions  # Use print_int 