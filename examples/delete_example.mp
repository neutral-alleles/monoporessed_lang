# Delete Example

# Assume print_int function

declare int32 temp_var
set temp_var 100
print_int temp_var # Output: 100

delete temp_var

# Using temp_var after deletion should result in an error 
# either during compilation (if statically checked) or runtime.
# print_int temp_var # ERROR: Unknown identifier 'temp_var'

print_int 5 # Just to show program continues


declare int32 outer_var
set outer_var 200

if true # Create a new scope
	declare int32 inner_var
	set inner_var 50
	print_int inner_var # Output: 50
	
	delete inner_var
	# print_int inner_var # ERROR: Unknown identifier 'inner_var'
	
	print_int outer_var # Output: 200 (outer scope still accessible)
	
# inner_var is out of scope here anyway, even without delete
# print_int inner_var # ERROR

print_int outer_var # Output: 200 