# Test control flow: if/elif/else, break, continue

init int32 value 15

# Test if/elif/else
if eq value 10
    print_int32 100 # Should not print
elif eq value 15
    print_int32 150 # Should print 150
else
    print_int32 200 # Should not print

print_int32 999 # Separator

# Test loop with break/continue
init int32 i 0 # Explicit iterator variable for while loop demo
while lt i 10
    # Use elif inside the loop too
    if eq i 2
        print_int32 222 # Indicate continue is hit
        inc i # Must increment before continue
        continue # Skip printing 2
    elif eq i 7
        print_int32 777 # Indicate break is hit
        break # Exit loop before printing 7
    else
        print_int32 i # Print other numbers

    inc i # Increment at the end of the loop body

# Expected output:
# 150
# 999
# 0
# 1
# 222  (continue hit)
# 3
# 4
# 5
# 6
# 777  (break hit) 