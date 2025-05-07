# Fibonacci Sequence Example

# Uses integer arithmetic keywords (add, sub, lt)
# Assumes print_int function
# Requires complex operands (nested ops/calls) to be parenthesized

def fib(int32 n) int32
	# Condition uses atomic operands - no parens needed
	if lte n 1 
		return n
	else 
		# Operands for set are complex (prefix op), require parens
		let int32 n_minus_1 (fib (sub n 1))
		let int32 n_minus_2 (fib (sub n 2))
		# Return expression involves nested calls/ops, requires parens
		return (add n_minus_1 n_minus_2)

# Calculate fibonacci of 10
let int32 n 40
# Operand for set is complex (function call), requires parens
let int32 result (fib n)

# Print the result
# Operand for print is complex (variable), but print is statement start, no parens needed.
print_int32 result
