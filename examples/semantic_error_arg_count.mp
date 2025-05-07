# Test Semantic Error: Incorrect Argument Count

def takes_one(int32 a) void
    return

# Error: Calling takes_one with 0 arguments
# takes_one

# Error: Calling takes_one with 2 arguments
takes_one 1 2 