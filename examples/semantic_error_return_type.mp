# Test Semantic Error: Incorrect Return Type

def get_number() int32
    # Error: Function declared to return int32, but returns bool
    return true 

# Main part (might not be reached if semantic error halts)
declare int32 x
set x (get_number)
print_int32 x 