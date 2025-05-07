# Rune Example

# Assumes print_rune function exists

declare rune initial
set initial 'V'
print_rune initial

declare rune newline_char
set newline_char '\n' # Escaped newline rune
print_rune newline_char

declare rune tab_char
set tab_char '\t'
print_rune tab_char

# Print a literal rune
print_rune '!'

print_rune initial
print_rune newline_char
print_rune tab_char 