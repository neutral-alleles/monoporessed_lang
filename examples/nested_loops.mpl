# Example of nested while loops using dec
declare int32 outer_count
declare int32 inner_count
set outer_count 3

while gt outer_count 0
    # Placeholder: Outer loop action (e.g., print outer_count)
    print_int outer_count # Use print_int
    set inner_count 5
    while gt inner_count 0
        # Placeholder: Inner loop action (e.g., print inner_count)
        print_int inner_count # Use print_int
        dec inner_count
    # Finished inner loop
    dec outer_count

# Example end state: outer_count = 0, inner_count = 0 (last value) 