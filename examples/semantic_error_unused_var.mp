# Test Semantic Error: Unused Variable

def my_func() void
    # declare int32 unused_local # Error: Declared but not used
    declare int32 used_local
    set used_local 10
    print_int32 used_local
    return

declare bool unused_global # Error: Declared but not used
declare bool used_global

set used_global true

# Call the function to analyze its scope
my_func

# Use the global variable
if used_global
    print_int32 1 