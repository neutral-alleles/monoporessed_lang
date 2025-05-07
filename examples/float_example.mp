# Floating Point Example

# Assumes print_float function exists

declare float64 pi
set pi 3.14159

declare float64 radius
set radius 5.0

# Calculate circumference: 2 * pi * radius
declare float64 circumference
set circumference fmul 2.0 pi # Need to handle multiplication by literal 2.0
set circumference fmul circumference radius
print_float circumference

# Calculate area: pi * radius * radius
declare float64 area
set area fmul pi radius
set area fmul area radius
print_float area

# Simple addition
declare float64 sum
set sum fadd 1.23 4.56
print_float sum 