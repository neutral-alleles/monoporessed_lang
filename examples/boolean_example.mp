# Boolean Logic Example

# Assumes print_bool function exists

declare bool flag1
set flag1 true
print_bool flag1 # true

declare bool flag2
set flag2 not flag1
print_bool flag2 # false

declare bool result_and
set result_and and flag1 flag2
print_bool result_and # false (true and false)

declare bool result_or
set result_or or flag1 flag2
print_bool result_or # true (true or false)

declare bool result_xor
set result_xor xor flag1 true # Use literal true
print_bool result_xor # false (true xor true)

# Chained operations (assuming parser handles precedence or structure)
# Example: not (flag1 and flag2)
declare bool temp_and
set temp_and and flag1 flag2
declare bool final_result
set final_result not (and flag1 flag2)
print_bool final_result # true

# Example with comparisons (assuming eq returns bool)
declare int32 x
declare int32 y
set x 5
set y 5
declare bool are_equal
set are_equal eq x y
print_bool are_equal # true

declare bool combined
set combined and are_equal (not flag2)
print_bool combined # true (true and (not false))

print_bool result_and
print_bool result_or
print_bool result_xor
print_bool final_result
print_bool combined 