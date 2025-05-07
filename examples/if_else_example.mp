# If-Else Example

# Assumes print_string function

declare int32 age
set age 25

declare bool is_adult
set is_adult gte age 18 # Assume gte (greater than or equal) exists

# Simple If
if is_adult
	print_string "User is an adult.\n"

# If-Else
declare int32 temperature
set temperature 15

if lt temperature 20 # Assume lt (less than) exists
	print_string "It\'s a bit chilly.\n"
else
	print_string "It\'s warm enough.\n"

# Nested If (example)
declare bool has_ticket
set has_ticket true

if is_adult
	if has_ticket
		print_string "Adult with ticket admitted.\n"
	else
		print_string "Adult needs a ticket.\n"
else
	print_string "Minor needs supervision (and maybe a ticket).\n"

declare string msg

if eq a b
	set msg "a is equal to b"
else:
	if lt a b
		set msg "a is less than b"
	else
		set msg "a is greater than b"

print_string msg # Use print_string 