# Factorial Example

# Uses integer arithmetic keywords (mul, add)
# Assumes print_int function

def factorial(int32 n) int32
	declare int32 result
	set result 1
	
	declare int32 i
	# Assuming range generates numbers from start up to (finish - 1)
	# Need expression evaluation for finish value
	declare int32 end_range
	set end_range add n 1 # Calculate n+1
	for i in range 1 end_range
		set result mul result i # Use 'mul' function style

	return result


declare int32 fact5
set fact5 factorial 5
print_int fact5 # Should print 120 

# Calculate factorial of 5
declare int32 n
declare int32 result
set n 5
set result factorial n

# Print the result
print_int result 