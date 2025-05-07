# Test type errors caught by semantic analysis

declare bool flag
declare int32 count
declare int32 result

init bool initial_flag true
init int32 initial_count 10

# set flag 5          # Error: Assigning int32 (5) to bool (flag)
# set count true      # Error: Assigning bool (true) to int32 (count)

# set result add flag count # Error: Using 'add' with bool and int32
# set flag and 1 0    # Error: Using 'and' with int32 operands

break # Error: break outside loop 