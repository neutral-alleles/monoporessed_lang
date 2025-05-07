# Example using inc/dec based on a condition
declare int32 score
set score 10
declare bool error_flag
set error_flag false # Assume some condition sets this later

# ... some code that might set error_flag based on input or calculation ...

print_int score # Use print_int

# Example condition check
if error_flag
    dec score
else
    inc score

print_int score # Use print_int

# Example end state: score will be 9 if error_flag was true, 11 otherwise 