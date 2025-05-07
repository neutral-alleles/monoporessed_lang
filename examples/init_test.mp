# Test the init statement

init int32 my_var add 5 3  # Initialize with an expression
init int32 another_var 10 # Initialize with a literal

print_int32 my_var        # Expected output: 8
print_int32 another_var   # Expected output: 10 