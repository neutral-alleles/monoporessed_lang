# Test Semantic Error: Incorrect Argument Type

def requires_bool(bool b) void
    # Function body doesn't matter for this test
    return

declare int32 my_int
set my_int 5

# Error: Calling requires_bool with int32 instead of bool
requires_bool my_int 