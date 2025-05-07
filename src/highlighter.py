# src/highlighter.py

import argparse
import sys
import logging

# Use relative imports assuming src is treated as a package
# Import the specific TT_ constants needed, not TokenType
from .lexer import (
    Lexer, TT_EOF, TT_NEWLINE, TT_INDENT, TT_DEDENT,
    TT_KEYWORD, TT_TYPE, TT_IDENTIFIER, TT_INTEGER, TT_FLOAT,
    TT_RUNE, TT_STRING, TT_BOOLEAN, TT_LPAREN, TT_RPAREN,
    TT_COMMA, TT_COLON, TT_COMMENT, TT_INVALID
)

# Configure basic logging for the highlighter (optional)
log = logging.getLogger('highlighter')
# Basic config if run standalone
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# --- Function to control lexer logging ---
def set_lexer_log_level(level):
    """Sets the logging level for the lexer's logger."""
    lexer_log = logging.getLogger('src.lexer') # Get the logger used in lexer.py
    lexer_log.setLevel(level)
    log.debug(f"Set src.lexer log level to {logging.getLevelName(level)}")

# --- ANSI Color Codes ---
# Basic colors (foreground)
COLOR_RESET = "\033[0m"
COLOR_BLACK = "\033[30m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"
COLOR_WHITE = "\033[37m"
# Bright versions
COLOR_BRIGHT_BLACK = "\033[90m" # Often used for comments (grey)
COLOR_BRIGHT_RED = "\033[91m"
COLOR_BRIGHT_GREEN = "\033[92m"
COLOR_BRIGHT_YELLOW = "\033[93m"
COLOR_BRIGHT_BLUE = "\033[94m"
COLOR_BRIGHT_MAGENTA = "\033[95m"
COLOR_BRIGHT_CYAN = "\033[96m"
COLOR_BRIGHT_WHITE = "\033[97m"

# --- Add Bold Code ---
BOLD = "\033[1m"
# --- Add Background Color for Limit ---
BG_BRIGHT_BLACK = "\033[100m" # Dark grey background

# --- Rainbow Colors for Parentheses ---
RAINBOW_COLORS = (
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_GREEN,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_MAGENTA,
)

# --- Token Type to Color Mapping ---
# Updated scheme for better contrast and emphasis
TOKEN_COLOR_MAP = {
    TT_KEYWORD: BOLD + COLOR_BRIGHT_BLUE,      # Bold Bright Blue
    TT_TYPE: BOLD + COLOR_BRIGHT_CYAN,       # Bold Bright Cyan
    TT_IDENTIFIER: COLOR_WHITE,                # White (Normal)
    TT_INTEGER: COLOR_BRIGHT_YELLOW,           # Bright Yellow
    TT_FLOAT: COLOR_BRIGHT_YELLOW,             # Bright Yellow
    TT_RUNE: COLOR_GREEN,                    # Green
    TT_STRING: COLOR_GREEN,                    # Green
    TT_BOOLEAN: COLOR_YELLOW,                  # Yellow
    TT_LPAREN: COLOR_BRIGHT_WHITE,           # Bright White
    TT_RPAREN: COLOR_BRIGHT_WHITE,           # Bright White
    TT_COMMA: COLOR_BRIGHT_WHITE,            # Bright White
    TT_COLON: COLOR_BRIGHT_WHITE,            # Bright White
    # Special tokens
    TT_NEWLINE: None, # Handled by reconstruction
    TT_INDENT: None, # Don't print any marker
    TT_DEDENT: None, # Don't print any marker
    TT_COMMENT: COLOR_BRIGHT_BLACK, # Color for comment text (now handled by SKIPPED_TEXT_COLOR)
    TT_INVALID: BOLD + COLOR_BRIGHT_RED,         # Bold Bright Red
    TT_EOF: None, # Handled by loop
}

DEFAULT_COLOR = COLOR_WHITE # Fallback for unmapped tokens (shouldn't be needed)
# --- Default color for skipped text (whitespace/comments) ---
# Set to None to use the default terminal text color
SKIPPED_TEXT_COLOR = None

# --- Mapping for specific keyword values ---
KEYWORD_VALUE_MAP = {
    'declare': BOLD + COLOR_BRIGHT_RED, # Use ANSI constants
    'set': BOLD + COLOR_BRIGHT_RED, # Use ANSI constants
    'let': BOLD + COLOR_BRIGHT_RED, # Use ANSI constants
    'init': BOLD + COLOR_BRIGHT_RED, # Use ANSI constants
}

def highlight_code(source_code: str):
    """Lexes the source code and prints it with ANSI syntax highlighting,
       reconstructing skipped whitespace and comments.
    """
    # --- Calculate padding for line numbers ---
    lines = source_code.count('\n') + 1 # Count lines
    line_num_padding = len(str(lines)) # Width needed for max line number

    lexer = Lexer(source_code)
    try:
        tokens = lexer.tokenize()
    except Exception as e:
        log.error(f"Lexer error: {e}")
        print(source_code) # Print raw code on lexer error
        return

    current_index = 0
    paren_depth = 0
    current_line_num = 1
    display_col = 0 # Track current display column
    limit = 79 # Column limit (0-indexed)

    # --- Helper to print text respecting the column limit --- 
    def print_limited(text, color):
        nonlocal display_col
        text_len = len(text)
        # Correct calculation based on 0-indexed limit and current display_col
        chars_before_limit = max(0, (limit + 1) - display_col)
        chars_after_limit = text_len - chars_before_limit

        # Print part before limit
        if chars_before_limit > 0:
            part_before = text[:chars_before_limit]
            if color:
                print(f"{color}{part_before}{COLOR_RESET}", end="")
            else:
                print(part_before, end="")
            display_col += len(part_before) # Update by visible length

        # Print part after limit with background color
        if chars_after_limit > 0:
            part_after = text[chars_before_limit:]
            # Apply background, then foreground (if any), then reset all
            if color:
                print(f"{BG_BRIGHT_BLACK}{color}{part_after}{COLOR_RESET}", end="")
            else:
                print(f"{BG_BRIGHT_BLACK}{part_after}{COLOR_RESET}", end="")
            display_col += len(part_after) # Update by visible length
    # --- End helper --- 

    # --- Print initial line number prefix and set initial display_col ---
    line_num_str = f"{current_line_num:>{line_num_padding}} | "
    prefix_width = len(line_num_str)
    if current_line_num % 5 == 0:
        print(f"{COLOR_GREEN}{line_num_str}{COLOR_RESET}", end="")
    else:
        print(line_num_str, end="")
    display_col = prefix_width

    for i, token in enumerate(tokens):
        # --- Process text leading up to the token ---
        if token.start_index > current_index:
            skipped_text = source_code[current_index:token.start_index]
            # Split skipped text only to handle newlines correctly
            parts = skipped_text.split('\n')
            for j, part in enumerate(parts):
                if j > 0: # Newline was encountered
                    print() # Print the newline
                    current_line_num += 1
                    # Print next line prefix and reset display_col
                    line_num_str = f"{current_line_num:>{line_num_padding}} | "
                    prefix_width = len(line_num_str)
                    if current_line_num % 5 == 0:
                        print(f"{COLOR_GREEN}{line_num_str}{COLOR_RESET}", end="")
                    else:
                        print(line_num_str, end="")
                    display_col = prefix_width
                
                # Print the text part using the limited printer
                if part: 
                    print_limited(part, None) # Skipped text has no foreground color

        # --- Process the token itself ---
        if token.type == TT_NEWLINE:
            print() # Print the newline character itself
            current_line_num += 1
            # Print next line prefix and reset display_col, peeking ahead
            if i + 1 < len(tokens) and tokens[i+1].type != TT_EOF:
                line_num_str = f"{current_line_num:>{line_num_padding}} | "
                prefix_width = len(line_num_str)
                if current_line_num % 5 == 0:
                    print(f"{COLOR_GREEN}{line_num_str}{COLOR_RESET}", end="")
                else:
                    print(line_num_str, end="")
                display_col = prefix_width
            else: # Last newline before EOF, just reset col
                display_col = 0
            current_index = token.end_index
            continue # Skip the rest of the loop for NEWLINE token

        if token.type == TT_EOF:
            break # Done processing

        # Determine color (Parentheses, Keywords, Identifiers)
        color = None
        # (Color determination logic remains the same)
        if token.type == TT_LPAREN:
            color_index = paren_depth % len(RAINBOW_COLORS)
            color = RAINBOW_COLORS[color_index]
            paren_depth += 1
        elif token.type == TT_RPAREN:
            paren_depth -= 1
            if paren_depth < 0:
                paren_depth = 0
                color = COLOR_BRIGHT_RED
            else:
                color_index = paren_depth % len(RAINBOW_COLORS)
                color = RAINBOW_COLORS[color_index]
        else:
            color = TOKEN_COLOR_MAP.get(token.type)
            if token.type == TT_KEYWORD:
                _color_override = None
                if token.value in KEYWORD_VALUE_MAP:
                    _color_override = KEYWORD_VALUE_MAP[token.value]
                elif isinstance(token.value, str) and token.value.startswith("read_"):
                    _color_override = COLOR_CYAN
                if _color_override:
                    color = _color_override
            elif token.type == TT_INVALID:
                color = TOKEN_COLOR_MAP[TT_INVALID]
            elif token.type == TT_IDENTIFIER:
                _color_override = None
                if isinstance(token.value, str) and token.value.startswith("print_"):
                    _color_override = COLOR_CYAN
                if _color_override:
                    color = _color_override
                else:
                    color = TOKEN_COLOR_MAP[TT_IDENTIFIER]

        # --- Print the token text using the limited printer ---
        token_text = source_code[token.start_index:token.end_index]
        if token_text: # Only print if token has text (e.g., not INDENT/DEDENT)
            print_limited(token_text, color)

        # Update the current index
        current_index = token.end_index

    # Add a final newline ONLY if the source didn't end with one.
    if not source_code.endswith('\n'):
        print()

def main():
    parser = argparse.ArgumentParser(description="Syntax highlight a code file using the lexer.")
    parser.add_argument("filename", help="Path to the code file to highlight.")
    parser.add_argument(
        "--lexer-log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING', # Set default level higher than DEBUG
        help="Set the logging level for the lexer module."
    )
    args = parser.parse_args()

    # --- SET LEXER LOG LEVEL BASED ON ARG ---
    log_level_name = args.lexer_log_level.upper()
    log_level = getattr(logging, log_level_name, logging.WARNING) # Default to WARNING if invalid
    set_lexer_log_level(log_level)

    try:
        with open(args.filename, 'r', encoding='utf-8') as f:
            source_code = f.read()
        log.info(f"Highlighting file: {args.filename}")
        highlight_code(source_code)
    except FileNotFoundError:
        log.error(f"Error: File not found: {args.filename}")
        sys.exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 