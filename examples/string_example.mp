# String Example

# Assumes print_string function exists

declare string greeting
set greeting "Hello"

declare string name
set name "Monopressed"

# We don't have concatenation yet, so just print parts
print_string greeting
print_string ", " # Print a literal string
print_string name
print_string "!\n"

# String with escapes
declare string escaped_str
set escaped_str "This string has a \"quote\" and a backslash \\ and a newline\nhere."
print_string escaped_str 