# Test mutability rules: let (immutable) and init (mutable)

let int32 immutable_val 10
init int32 mutable_val 20

print_int32 immutable_val # Expected: 10
print_int32 mutable_val   # Expected: 20

# This should be allowed
set mutable_val 30 
print_int32 mutable_val   # Expected: 30

# This should cause a SemanticError
set immutable_val 40 

print_int32 999 # Separator

# Test assignment within expression evaluation (should be okay)
let int32 another_immutable (add immutable_val 5)
init int32 another_mutable (add mutable_val 5)

print_int32 another_immutable # Expected: 15
print_int32 another_mutable   # Expected: 35 