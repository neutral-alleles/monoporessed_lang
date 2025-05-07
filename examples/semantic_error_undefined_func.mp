# Test Semantic Error: Undefined Function Call

declare int32 result

# Error: Calling a function that was never defined
set result (call_something 10)

print_int32 result 