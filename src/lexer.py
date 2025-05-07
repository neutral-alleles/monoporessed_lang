import re
from collections import namedtuple
import logging
import string
from typing import List, Optional
from dataclasses import dataclass
from typing import Any

# Basic Logging Setup (can be configured further)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)

@dataclass
class Token:
    type: str
    value: Any
    line: int  # 1-based line number
    col: int   # 1-based column number (start of token)
    start_index: int # Index in the original text where the token starts
    end_index: int   # Index in the original text where the token ends (exclusive)

    def __repr__(self):
        # Make EOF/Newline/Indent/Dedent cleaner in logs
        if self.type in [TT_EOF, TT_NEWLINE, TT_INDENT, TT_DEDENT]:
            return f"{self.type}(@{self.line}:{self.col} [{self.start_index}:{self.end_index}])"
        return f"{self.type}({repr(self.value)} @{self.line}:{self.col} [{self.start_index}:{self.end_index}])"

class IndentationError(Exception):
    pass

# Token Types
TT_KEYWORD = "KEYWORD"
TT_TYPE = "TYPE"
TT_IDENTIFIER = "IDENTIFIER"
TT_INTEGER = "INTEGER"
TT_FLOAT = "FLOAT"
TT_BOOLEAN = "BOOLEAN"
TT_STRING = "STRING"
TT_RUNE = "RUNE"
TT_LPAREN = "LPAREN"    # (
TT_RPAREN = "RPAREN"    # )
TT_COMMA = "COMMA"      # ,
TT_COLON = "COLON"      # :
TT_NEWLINE = "NEWLINE"  # \n
TT_INDENT = "INDENT"    # Increased indentation
TT_DEDENT = "DEDENT"    # Decreased indentation
TT_COMMENT = "COMMENT"    # #...
TT_EOF = "EOF"        # End Of File
TT_INVALID = "INVALID"  # Unrecognized token

# Define keywords (used by both lexer and parser for consistency)
KEYWORDS = {
    'set', 'return', 'if', 'else', 'while', 'for', 'in', 'range',
    # Function related
    'def',
    # Arithmetic
    'add', 'sub', 'mul', 'div',
    'fadd', 'fsub', 'fmul', 'fdiv',
    # Comparison
    'eq', 'neq', 'lt', 'lte', 'gt', 'gte',
    # Logical
    'not', 'and', 'or',
    # Bitwise
    'xor',
    # Other
    'inc', 'dec', 
    'selfset',
    # IO
    'read_int32', 'read_int64', 'read_float32', 'read_float64', 'read_string', 'read_rune',
    # Combined Decl/Set
    'init', # Mutable variable definition
    'let',  # Immutable variable definition
    # Control Flow Additions
    'break', 'continue', 'elif'
}

TYPES = {
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
    'float32', 'float64', 'bool', 'rune', 'string', 'unit', 'void'
}

# Combine all token types (useful for debugging/logging if needed)
TOKEN_TYPES = {
    TT_INTEGER, TT_FLOAT, TT_BOOLEAN, TT_RUNE, TT_STRING,
    TT_IDENTIFIER, TT_KEYWORD, TT_TYPE,
    TT_LPAREN, TT_RPAREN, TT_COMMA, TT_COLON,
    TT_NEWLINE, TT_INDENT, TT_DEDENT,
    TT_EOF, TT_INVALID
}

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1 # 1-based column
        self.tokens = []
        self.indent_stack = [0] # Stack of indentation levels (equivalent spaces)
        self.indent_size = 4 # Standard indent size (treat tabs as 4 spaces)
        self.at_line_start = True # Flag to track if we are at the start of a new line
        self._current_line_char_count = 0 # Track character count of the current line
        log.debug(f"Lexer initialized with text of length {len(text)}")

    def _advance(self, count=1):
        """Advance position and column, handling newlines and line length check."""
        for _ in range(count):
            if self.pos < len(self.text):
                char = self.text[self.pos]
                # --- Line Length Check --- 
                if char == '\n':
                    if self._current_line_char_count > 79:
                        # Warning for the line that just ended
                        log.warning(f"Source line {self.line} exceeds 79 characters ({self._current_line_char_count}).")
                    self._current_line_char_count = 0 # Reset for next line
                    self.line += 1
                    self.col = 1
                else:
                    self._current_line_char_count += 1 # Increment char count
                    self.col += 1
                # --- End Check --- 
                self.pos += 1 # Advance position *after* checking char
            else:
                break

    def _peek(self, lookahead=0):
        """Return the character at pos + lookahead without consuming it, or None if EOF."""
        peek_pos = self.pos + lookahead
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        return None

    def _skip_whitespace_and_comments(self):
        """Skips spaces and comments. Does NOT handle indentation (tabs/newlines)."""
        skipped = False
        while self._peek() is not None:
            char = self._peek()
            # Skip only spaces, not newline or tab 
            if char == ' ': # Check explicitly for space
                self._advance()
                skipped = True
            # Skip # comments
            elif char == '#':
                log.debug(f"[LEXER] Skipping comment at {self.line}:{self.col}")
                start_comment_line = self.line
                while self._peek() is not None and self._peek() != '\n':
                    self._advance()
                # Don't advance past newline here; let the main loop handle it or EOF.
                log.debug(f"[LEXER] Comment ended at line {start_comment_line}. Current pos at {self.line}:{self.col} (char: {repr(self._peek())})")
                skipped = True # We skipped something
            else:
                break # Stop if not space or comment start
        return skipped

    def tokenize(self):
        log.debug("Starting tokenization...")
        while self.pos < len(self.text):
            start_pos = self.pos # Record position at start of loop iteration
            start_line = self.line
            start_col = self.col

            # Check indentation at the start of a line
            if self.at_line_start:
                log.debug(f"At line start {self.line}:{self.col}. Checking indentation.")
                start_indent_col = self.col
                start_index = self.pos
                leading_spaces = 0
                leading_tabs = 0

                # Consume leading whitespace
                while True:
                    char_at_pos = self.text[self.pos] if self.pos < len(self.text) else None
                    # --- DEBUG LOG ADDED ---
                    char_ord = ord(char_at_pos) if char_at_pos else -1
                    log.debug(f"    Indent loop check: char={repr(char_at_pos)}, ord={char_ord}, pos={self.pos}")
                    # --- END DEBUG LOG ---
                    if char_at_pos == ' ':
                        leading_spaces += 1
                        self.pos += 1
                    elif char_ord == 9: # Check explicitly using ord(TAB) == 9
                        leading_tabs += 1
                        self.pos += 1
                    else:
                        log.debug(f"    Indent loop break: char={repr(char_at_pos)}, ord={char_ord}") # Log break reason
                        break # End of leading whitespace

                # Update column based on actual characters consumed
                consumed_chars = self.pos - start_index
                self.col = start_indent_col + consumed_chars

                # Validate mixing tabs and spaces
                if leading_spaces > 0 and leading_tabs > 0:
                    log.error(f"Indentation Error: Mixed spaces and tabs on line {self.line}.")
                    raise IndentationError(f"Mixed spaces and tabs are not allowed at line {self.line}")

                # Calculate equivalent spaces
                num_spaces = 0
                if leading_tabs > 0:
                    num_spaces = leading_tabs * self.indent_size
                elif leading_spaces > 0:
                    num_spaces = leading_spaces
                # else: num_spaces remains 0 if no indentation

                # Validate indentation level consistency (only if spaces were used)
                if leading_spaces > 0 and num_spaces % self.indent_size != 0:
                     log.error(f"Indentation Error: Found {num_spaces} spaces, not a multiple of {self.indent_size}.")
                     raise IndentationError(f"Invalid indentation level ({num_spaces} spaces) at line {self.line}")

                current_level = num_spaces // self.indent_size

                log.debug(f"Finished indent check. Spaces={leading_spaces}, Tabs={leading_tabs}. Equivalent={num_spaces} (Level {current_level}). Actual Col: {self.col}. Stack: {self.indent_stack}.")

                # --- Check if line is blank/comment *before* comparing levels ---
                peek_char_after_indent = self._peek() # Peek after consuming indent whitespace
                is_blank_or_comment_line = False
                if peek_char_after_indent == '\n' or peek_char_after_indent is None or peek_char_after_indent == '#':
                    is_blank_or_comment_line = True
                    # Handle comment skipping if needed (doesn't change blank status)
                    if peek_char_after_indent == '#':
                        log.debug(f"Skipping comment on potentially blank line after indent check at {self.line}:{self.col}")
                        # Use _advance here as _peek was already done
                        self._advance() # Consume '#'
                        while self._peek() is not None and self._peek() != '\n':
                            self._advance()
                        peek_char_after_indent = self._peek() # Re-peek after comment
                    
                    # Handle newline or EOF after indent/comment
                    if peek_char_after_indent == '\n':
                         log.debug(f"Skipping newline of blank/comment-only line (indent level {current_level} ignored).")
                         self._advance() # Consume the newline
                         self.at_line_start = True # Immediately ready for next line's indent
                         # Do NOT compare levels or emit INDENT/DEDENT for blank/comment lines
                         continue # Continue to next line
                    elif peek_char_after_indent is None: # EOF right after indentation/comment
                        log.debug("EOF encountered directly after indentation check or comment on blank line. Breaking loop.")
                        # Do NOT compare levels or emit INDENT/DEDENT
                        break # Let EOF handling take over
                
                # --- Compare Levels only if line has content ---
                if not is_blank_or_comment_line:
                    indent_change_emitted = False # Track if INDENT/DEDENT was emitted
                    indent_token_pos = self.pos # Position *after* consuming indent whitespace
                    if current_level > self.indent_stack[-1]:
                        # Must be exactly one level higher
                        if current_level != self.indent_stack[-1] + 1:
                             log.error(f"Indentation Error: Unexpected indent level increase from {self.indent_stack[-1]} to {current_level}")
                             raise IndentationError(f"Invalid indentation increase at line {self.line}")
                        log.debug("INDENT detected.")
                        # Indent/Dedent have zero length at the point they occur
                        indent_token = Token(TT_INDENT, None, self.line, start_indent_col, indent_token_pos, indent_token_pos)
                        log.debug(f"--> Appending INDENT token: {indent_token}")
                        self.tokens.append(indent_token)
                        self.indent_stack.append(current_level)
                        indent_change_emitted = True
                        log.debug(f"<-- Appended INDENT. Tokens len: {len(self.tokens)}, Last: {self.tokens[-1] if self.tokens else 'None'}")
    
                    elif current_level < self.indent_stack[-1]:
                        while current_level < self.indent_stack[-1]:
                            log.debug("DEDENT detected.")
                            # Indent/Dedent have zero length at the point they occur
                            dedent_token = Token(TT_DEDENT, None, self.line, start_indent_col, indent_token_pos, indent_token_pos)
                            log.debug(f"--> Appending DEDENT token: {dedent_token}")
                            self.tokens.append(dedent_token)
                            self.indent_stack.pop()
                            indent_change_emitted = True
                            log.debug(f"<-- Appended DEDENT. Tokens len: {len(self.tokens)}, Last: {self.tokens[-1] if self.tokens else 'None'}")
                        # After popping, the current level must match the new top of the stack
                        if current_level != self.indent_stack[-1]:
                            log.error(f"Inconsistent dedentation. Expected indent level {self.indent_stack[-1]}, but calculated {current_level} after dedenting.")
                            raise IndentationError(f"Inconsistent dedentation at line {self.line}")
                    # else: current_level == self.indent_stack[-1] -> no change needed
    
                    self.at_line_start = False # Indentation handled for this line
    
                    # --- REMOVED: Blank line handling moved earlier ---
                    # --- REMOVED: Continue if INDENT/DEDENT emitted (parser handles this) ---
                    # if indent_change_emitted:
                    #      log.debug("Continuing loop after emitting INDENT/DEDENT on a content line.")
                    #      continue # Let the next loop iteration handle the content
                # else (is_blank_or_comment_line): Already handled by the continue/break above

            # --- End Indentation Handling ---

            # Skip non-newline/tab whitespace and comments AFTER indentation check
            # --- REVERTING: Capture start position *after* skipping again ---
            skipped_intra_line = self._skip_whitespace_and_comments()
            if skipped_intra_line:
                 log.debug(f"Skipped intra-line whitespace/comment. Current pos: {self.line}:{self.col}")

            # Capture start position *after* potential skipping for the token itself
            start_line, start_col = self.line, self.col
            start_index = self.pos

            # Re-peek after potential skipping
            char = self._peek()

            # Handle EOF after potential skipping
            if char is None:
                log.debug("Reached EOF after skipping.")
                break

            # Log the character we are actually processing now
            log.debug(f"Processing character '{repr(char)}' at {start_line}:{start_col} (Index {start_index})")

            # Newline
            if char == '\n':
                log.debug("Found NEWLINE")
                # Length check is now done within _advance
                token_start_index = start_index
                token_start_col = start_col
                # Need current line *before* _advance increments it in the next loop
                line_num_for_token = self.line 
                # Call _advance here ONLY to move position, line/col/count handled inside
                self._advance()
                token_end_index = self.pos
                self.tokens.append(Token(TT_NEWLINE, '\n', line_num_for_token, token_start_col, token_start_index, token_end_index))
                self.at_line_start = True
                continue
            
            # Tab - REMOVED - Should not be encountered here if indentation logic is correct
            # if char == '\t':
            #     log.error(f"Unexpected TAB encountered at {start_line}:{start_col} outside of line start. Lexer bug?")
            #     self.tokens.append(Token(TT_INVALID, '\t', start_line, start_col))
            #     self._advance()
            #     continue 

            # Punctuation
            if char in '(),:':
                log.debug(f"Found PUNCTUATION: {char}")
                token_type = {
                    '(': TT_LPAREN, ')': TT_RPAREN,
                    ',': TT_COMMA, ':': TT_COLON,
                }[char]
                token_start_index = start_index
                self._advance()
                token_end_index = self.pos
                self.tokens.append(Token(token_type, char, start_line, start_col, token_start_index, token_end_index))
                continue

            # Numbers (Float/Integer)
            # Modified to handle optional leading negative sign
            is_negative = False
            if char == '-' and self._peek(1) and (self._peek(1).isdigit() or self._peek(1) == '.'):
                log.debug("Possible NEGATIVE NUMBER start")
                is_negative = True
                self._advance() # Consume '-'
                char = self._peek() # Update char to the digit/dot after '-'
                if char is None: # Handle case like just "-"
                    log.error(f"Invalid token: isolated '-' at {start_line}:{start_col}")
                    token_start_index = start_index
                    # self._advance() was already called
                    token_end_index = self.pos
                    self.tokens.append(Token(TT_INVALID, '-', start_line, start_col, token_start_index, token_end_index))
                    continue
            
            if char is not None and (char.isdigit() or (char == '.' and self._peek(1) and self._peek(1).isdigit())):
                log.debug(f"Starting number parse (is_negative={is_negative}) at {self.line}:{self.col}")
                num_str = ""
                has_decimal = False
                # Capture the starting position of the number string itself
                num_start_index = self.pos if not is_negative else self.pos -1 

                while True:
                    current_char = self._peek()
                    if current_char == '.':
                        if has_decimal: break # Second decimal point
                        # Allow dot even if not followed by digit (e.g., 10.)
                        has_decimal = True
                        num_str += current_char
                        self._advance()
                    elif current_char is not None and current_char.isdigit():
                        num_str += current_char
                        self._advance()
                    else:
                        break # End of number (whitespace, operator, EOF, etc.)
                
                log.debug(f"Parsed number string content: '{num_str}', has_decimal={has_decimal}")

                # If it ends with a dot, treat as float even if no digits followed
                if num_str.endswith('.'):
                    has_decimal = True 
                    # Append a '0' conceptually for float conversion, but keep original string for errors

                # Validation and Token Creation
                if not num_str or num_str == '.': # Invalid if empty or just '.' after potential '-'
                     final_num_str_for_error = self.text[num_start_index:self.pos] # Get original slice
                     log.warning(f"Invalid number format: '{final_num_str_for_error}' at {start_line}:{start_col}. Treating as invalid.")
                     # Use num_start_index which includes potential '-' sign
                     self.tokens.append(Token(TT_INVALID, final_num_str_for_error, start_line, start_col, num_start_index, self.pos))
                     continue

                final_num_str = ('-' + num_str) if is_negative else num_str
                
                if has_decimal:
                    try:
                        # Ensure conversion works even if ending with '.' by adding '0'
                        f_val = float(final_num_str + '0' if final_num_str.endswith('.') else final_num_str)
                        log.debug(f"Found FLOAT: {f_val}")
                        # Use num_start_index which includes potential '-' sign
                        self.tokens.append(Token(TT_FLOAT, f_val, start_line, start_col, num_start_index, self.pos))
                    except ValueError:
                        log.error(f"Invalid float literal '{final_num_str}' at {start_line}:{start_col}")
                        self.tokens.append(Token(TT_INVALID, final_num_str, start_line, start_col, num_start_index, self.pos))
                else: # Integer
                    try:
                        i_val = int(final_num_str)
                        log.debug(f"Found INTEGER: {i_val}")
                        # Use num_start_index which includes potential '-' sign
                        self.tokens.append(Token(TT_INTEGER, i_val, start_line, start_col, num_start_index, self.pos))
                    except ValueError: # Should not happen if checks are correct, but safeguard
                        log.error(f"Invalid integer literal '{final_num_str}' at {start_line}:{start_col}")
                        self.tokens.append(Token(TT_INVALID, final_num_str, start_line, start_col, num_start_index, self.pos))
                continue # Continue main loop after number

            # Identifiers, Keywords, Types, Booleans
            if char.isalpha():
                log.debug("Possible IDENTIFIER/KEYWORD/TYPE/BOOL start")
                ident_str = ""
                while True:
                    current_char = self._peek()
                    # Allow letters, numbers, or underscores after the first char
                    if current_char is None or not (current_char.isalnum() or current_char == '_'):
                        break
                    ident_str += current_char
                    self._advance()
                
                log.debug(f"Parsed identifier-like string: '{ident_str}'")
                
                # Check for Booleans FIRST
                if ident_str == 'true':
                    log.debug(f"Found BOOLEAN: True")
                    self.tokens.append(Token(TT_BOOLEAN, True, start_line, start_col, start_index, self.pos))
                elif ident_str == 'false':
                    log.debug(f"Found BOOLEAN: False")
                    self.tokens.append(Token(TT_BOOLEAN, False, start_line, start_col, start_index, self.pos))
                # Then check for Keywords
                elif ident_str in KEYWORDS:
                    log.debug(f"Found KEYWORD: {ident_str}")
                    self.tokens.append(Token(TT_KEYWORD, ident_str, start_line, start_col, start_index, self.pos))
                # Then check for Types
                elif ident_str in TYPES:
                    log.debug(f"Found TYPE: {ident_str}")
                    self.tokens.append(Token(TT_TYPE, ident_str, start_line, start_col, start_index, self.pos))
                # Otherwise, it's an Identifier
                else:
                    log.debug(f"Found IDENTIFIER: {ident_str}")
                    self.tokens.append(Token(TT_IDENTIFIER, ident_str, start_line, start_col, start_index, self.pos))
                continue

            # String Literals (")
            if char == '"':
                log.debug("Possible STRING start")
                str_val = ""
                self._advance() # Consume opening quote
                string_terminated = False
                while self._peek() is not None:
                    current_char = self._peek()
                    char_line, char_col = self.line, self.col # Position of char itself

                    if current_char == '"':
                        self._advance() # Consume closing quote
                        log.debug(f"Found STRING (terminated): '{str_val}'")
                        self.tokens.append(Token(TT_STRING, str_val, start_line, start_col, start_index, self.pos))
                        string_terminated = True
                        break
                    elif current_char == '\\':
                        self._advance() # Consume backslash
                        escape_char = self._peek()
                        if escape_char is None:
                           log.error(f"Unterminated escape sequence at EOF in string starting {start_line}:{start_col}")
                           # Invalid token spans from opening quote to current position (before EOF)
                           self.tokens.append(Token(TT_INVALID, self.text[start_index:self.pos], start_line, start_col, start_index, self.pos))
                           string_terminated = True # Mark as error-terminated
                           break
                        
                        self._advance() # Consume escape character
                        if escape_char == 'n': str_val += '\n'
                        elif escape_char == 't': str_val += '\t'
                        elif escape_char == '\\': str_val += '\\'
                        elif escape_char == '"': str_val += '"'
                        else:
                            log.warning(f"Unknown string escape sequence '\\{escape_char}' at {char_line}:{char_col}. Treating literally.")
                            str_val += '\\' + escape_char
                    elif current_char == '\n':
                        log.error(f"Unterminated string literal (newline encountered) starting {start_line}:{start_col}")
                        # Invalid token spans from opening quote to current position (before newline)
                        self.tokens.append(Token(TT_INVALID, self.text[start_index:self.pos], start_line, start_col, start_index, self.pos))
                        string_terminated = True # Mark as error-terminated
                        # DO NOT CONSUME the newline. Let the main loop handle it.
                        break
                    else:
                        str_val += current_char
                        self._advance()
                
                if not string_terminated:
                     log.error(f"Unterminated string literal at EOF starting {start_line}:{start_col}")
                     # Invalid token spans from opening quote to end of text
                     self.tokens.append(Token(TT_INVALID, self.text[start_index:self.pos], start_line, start_col, start_index, self.pos))
                continue

            # Rune Literals (')
            if char == "'":
                start_rune_line, start_rune_col = start_line, start_col # Use these for ALL rune-related tokens
                start_rune_index = self.pos # **** CAPTURE START INDEX ****
                log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] START (Index {start_rune_index}). Consuming opening quote.")
                self._advance() # Consume opening quote

                rune_val = None
                token_emitted = False

                # **** REVISED: Check for Empty or EOF FIRST ****
                first_char = self._peek()
                if first_char == "'": # Check for empty ''
                    log.error(f"[RUNE @{start_rune_line}:{start_rune_col}] ERROR: Empty rune literal (''). Consuming closing quote.")
                    self._advance() # Consume closing quote
                    # Invalid token is '' which has length 2
                    self.tokens.append(Token(TT_INVALID, "''", start_rune_line, start_rune_col, start_rune_index, self.pos))
                    token_emitted = True
                elif first_char is None: # Check for EOF immediately after '
                    log.error(f"[RUNE @{start_rune_line}:{start_rune_col}] ERROR: Unterminated rune literal (') at EOF.")
                    # Don't consume EOF
                    # Invalid token is just ' which has length 1
                    self.tokens.append(Token(TT_INVALID, "'", start_rune_line, start_rune_col, start_rune_index, self.pos))
                    token_emitted = True
                else:
                    # **** Proceed to process content (Step 2 & 3) ****
                    char1 = first_char # Use the peeked char
                    log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Peek char1: {repr(char1)}")

                    # 2. Process potential rune character
                    log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Processing char1 {repr(char1)}.")
                    # **** Handle Raw vs Normal ****
                    if char1 is not None and ord(char1) == ord('\\'): # Escape sequence 
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Found escape sequence start '\\'.")
                        escape_start_line, escape_start_col = self.line, self.col # Pos of the \\
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Consuming backslash. Pos before: {self.pos}")
                        self._advance() # Consume backslash
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Pos after consuming backslash: {self.pos}")
                        char2 = self._peek()
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Peek char2 (after \\): {repr(char2)}")

                        if char2 is None: # EOF after \\
                            log.error(f"[RUNE @{start_rune_line}:{start_rune_col}] ERROR: Unterminated rune escape sequence ('\\') at EOF.")
                            # Invalid segment is '\
                            self.tokens.append(Token(TT_INVALID, self.text[start_rune_index:self.pos], start_rune_line, start_rune_col, start_rune_index, self.pos))
                            token_emitted = True
                        else: # Process escape char
                            # **** ADDED LOGGING ****
                            log.debug(f"[RUNE ESC @{escape_start_line}:{escape_start_col}] BEFORE consuming char2={repr(char2)}. Pos={self.pos}, Line={self.line}, Col={self.col}, NextChar={repr(self._peek())}")
                            self._advance() # Consume the char after backslash (char2)
                            log.debug(f"[RUNE ESC @{escape_start_line}:{escape_start_col}] AFTER consuming char2. Pos={self.pos}, Line={self.line}, Col={self.col}, NextChar={repr(self._peek())}")
                            # **** END LOGGING ****

                            # Determine rune value from escape char
                            if char2 == 'n': rune_val = '\n'
                            elif char2 == 't': rune_val = '\t'
                            elif char2 == '\\': rune_val = '\\'
                            elif char2 == "'": rune_val = "'"
                            else: rune_val = None # Invalid escape

                            # **** REVISED ESCAPE LOGIC ****
                            if rune_val is not None: # Valid escape sequence
                                log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Parsed valid escape sequence yielding {repr(rune_val)}.")
                                # Check for the closing quote IMMEDIATELY after the escape sequence
                                if self._peek() == "'":
                                    log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Found closing quote after escape. Consuming.")
                                    self._advance() # Consume closing quote
                                    # Valid rune spans from opening quote to closing quote
                                    self.tokens.append(Token(TT_RUNE, rune_val, start_rune_line, start_rune_col, start_rune_index, self.pos))
                                    token_emitted = True
                                else: # Missing closing quote after valid escape
                                    log.error(f"[RUNE @{start_rune_line}:{start_rune_col}] ERROR: Unterminated rune literal after escape sequence '\\{char2}'. Expected closing quote, found {repr(self._peek())}.")
                                    # The invalid segment is the opening quote + escape sequence
                                    invalid_segment = self.text[start_rune_index : self.pos] # e.g., "'\\n"
                                    self.tokens.append(Token(TT_INVALID, invalid_segment, start_rune_line, start_rune_col, start_rune_index, self.pos))
                                    token_emitted = True
                            else: # Invalid escape sequence like \z
                                log.error(f"[RUNE @{start_rune_line}:{start_rune_col}] ERROR: Invalid rune escape sequence '\\{char2}' at {escape_start_line}:{escape_start_col}.")
                                invalid_segment = self.text[start_rune_index : self.pos] # e.g., "'\\z"
                                # Check if closing quote follows immediately
                                if self._peek() == "'":
                                    log.debug(f"[RUNE ERROR FIX @{start_rune_line}:{start_rune_col}] Consuming closing quote after invalid escape.")
                                    self._advance() # Consume closing quote to include it in the invalid span
                                    invalid_segment = self.text[start_rune_index : self.pos] # e.g., "'\\z'"
                                self.tokens.append(Token(TT_INVALID, invalid_segment, start_rune_line, start_rune_col, start_rune_index, self.pos))
                                token_emitted = True

                    else: # Normal character
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Processing char1 {repr(char1)} as normal character.")
                        # It's a regular character like 'a' or any char in raw mode
                        rune_val = char1
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Consuming normal char {repr(char1)}. Pos before: {self.pos}")
                        self._advance() # Consume the rune character (char1)
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Pos after consuming normal char: {self.pos}. rune_val={repr(rune_val)}")

                # 3. Check for closing quote if no token emitted yet
                #    (This runs after processing escape OR normal char if no error occurred)
                # **** ADDED LOGGING ****
                log.debug(f"[RUNE STEP 3 @{start_rune_line}:{start_rune_col}] Pre-check state: token_emitted={token_emitted}, rune_val={repr(rune_val)}, Pos={self.pos}, Line={self.line}, Col={self.col}, NextChar={repr(self._peek())}")
                # **** END LOGGING ****
                if not token_emitted:
                    char_after = self._peek()
                    log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Peek char_after: {repr(char_after)}")
                    if char_after == "'": # Found closing quote - SUCCESS!
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Found closing quote. Consuming it. Pos before: {self.pos}")
                        self._advance() # Consume closing quote
                        log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] Pos after consuming closing quote: {self.pos}")
                        if rune_val is not None: # Should always be true unless internal error
                             log.debug(f"[RUNE @{start_rune_line}:{start_rune_col}] SUCCESS: Found RUNE: '{rune_val}'")
                             # Valid rune spans from opening quote to closing quote
                             self.tokens.append(Token(TT_RUNE, rune_val, start_rune_line, start_rune_col, start_rune_index, self.pos))
                        else:
                             # Should not happen if code path is correct
                             log.error(f"[RUNE @{start_rune_line}:{start_rune_col}] INTERNAL ERROR: Reached valid rune end but rune_val is None. Pos:{self.pos}")
                             self.tokens.append(Token(TT_INVALID, self.text[start_rune_index:self.pos], start_rune_line, start_rune_col, start_rune_index, self.pos))
                        token_emitted = True # Mark as handled

                    else: # Error: Missing closing quote or too many chars (e.g., 'ab', 'a<EOF>')
                        # State: self.pos points to the character *inside* the quote that should be followed by '
                        # Example 'a : pos points at 'a'
                        # Example 'ab': pos points at 'a' (if logic advances char by char) - NO, current code advances *after* reading 'a', so pos points at 'b'
                        # Let's assume pos points at the character *after* the intended single rune character.
                        # Example 'a<EOF>': pos points at 'a'. char_after is None.
                        # Example 'ab': pos points at 'a'. char_after is 'b'.
                        # Example 'ab'c: pos points at 'a'. char_after is 'b'.

                        # **** REVISED ERROR LOGIC ****
                        # current_char = self.text[self.pos] # The character we just consumed (e.g., 'a' in 'a', 'a' in 'ab') -> This caused IndexError if pos advanced past end.
                        # Let's get the character *before* the current position for logging/segment creation.
                        consumed_char = self.text[self.pos-1] if self.pos > start_rune_index + 1 else '' # Get char if we consumed one

                        log_msg = f"[RUNE @{start_rune_line}:{start_rune_col}] ERROR: Invalid rune. "
                        invalid_segment = ""

                        if char_after is None: # Unterminated after one char (e.g., 'a<EOF>)
                            log_msg += f"Unterminated after character {repr(consumed_char)}. Reached EOF."
                            # The invalid segment is the opening quote and the single char
                            # Use start_rune_index and current self.pos which is after the consumed char.
                            # **** ADJUST INVALID VALUE to match test ****
                            invalid_segment = self.text[start_rune_index : self.pos] + "EOF..." # e.g. "'aEOF..."
                            # No more advances needed.
                            # The invalid token spans from start quote to current pos (end of text)
                            self.tokens.append(Token(TT_INVALID, self.text[start_rune_index:self.pos], start_rune_line, start_rune_col, start_rune_index, self.pos))
                            token_emitted = True

                        else: # Too long (e.g., 'ab', 'a?') char_after is not \"'\"
                            log_msg += f"Expected closing quote after {repr(consumed_char)}, but found {repr(char_after)}."
                            invalid_segment = self.text[start_rune_index : self.pos + 1] + "..." # e.g. \"'ab...\"
                            log.debug(f"[RUNE ERROR @{start_rune_line}:{start_rune_col}] Consuming unexpected char {repr(char_after)}.")
                            self._advance() # Consume char_after (e.g. 'b')

                            # Check if closing quote follows the invalid extra char
                            if self._peek() == "'":
                                 log.debug(f"[RUNE ERROR @{start_rune_line}:{start_rune_col}] Consuming closing quote after invalid sequence.")
                                 self._advance() # Consume closing quote to include it
                                 invalid_segment = self.text[start_rune_index : self.pos] # e.g., \"'ab'\"
                        log.error(log_msg)
                        self.tokens.append(Token(TT_INVALID, invalid_segment, start_rune_line, start_rune_col, start_rune_index, self.pos))
                        token_emitted = True

                # **** ADD LOGGING ****
                log.debug(f"[RUNE END @{start_rune_line}:{start_rune_col}] Checking if token emitted: {token_emitted}")
                if token_emitted:
                     log.debug(f"[RUNE END @{start_rune_line}:{start_rune_col}] Token emitted. State BEFORE skip: Pos={self.pos}, Line={self.line}, Col={self.col}, NextChar={repr(self._peek())}")
                     skipped_after = self._skip_whitespace_and_comments()
                     log.debug(f"[RUNE END @{start_rune_line}:{start_rune_col}] Token emitted. State AFTER skip (skipped={skipped_after}): Pos={self.pos}, Line={self.line}, Col={self.col}, NextChar={repr(self._peek())}")
                     log.debug(f"[RUNE END @{start_rune_line}:{start_rune_col}] Continuing to next loop iteration.")
                     continue

            # Invalid character (if nothing else matched)
            log.error(f"Invalid character '{repr(char)}' at {start_line}:{start_col}")
            token_start_index = start_index
            self._advance()
            token_end_index = self.pos
            self.tokens.append(Token(TT_INVALID, char, start_line, start_col, token_start_index, token_end_index))

        # --- EOF Handling --- 
        # Check length of the very last line
        if self._current_line_char_count > 79:
             log.warning(f"Source line {self.line} exceeds 79 characters ({self._current_line_char_count}).")
        # --- End Check ---

        # Emit DEDENT tokens for any remaining indentation levels
        log.debug(f"Reached end of text. Final indent stack: {self.indent_stack}")
        final_eof_pos = self.pos # Position at the very end of the text
        while self.indent_stack[-1] > 0:
            log.debug("Emitting DEDENT at EOF.")
            self.indent_stack.pop()
            # DEDENTs at EOF have zero length at the final position
            self.tokens.append(Token(TT_DEDENT, None, self.line, self.col, final_eof_pos, final_eof_pos))

        # Construct position string directly for the log message
        log.debug(f"Appending EOF token at @{self.line}:{self.col}")
        # EOF has zero length at the final position
        self.tokens.append(Token(TT_EOF, None, self.line, self.col, final_eof_pos, final_eof_pos))
        log.debug("Tokenization finished.")
        return self.tokens

# Example Usage (for testing)
if __name__ == "__main__":
    code = """
declare int32 counter
set counter 0

def add(int32 a, int32 b) int32
	# This is a comment
	declare int32 result
	set result a # Not quite right syntax yet
	return result

# Main part
set counter add counter 1
while false:
	set counter 0
"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    for token in tokens:
        print(token) 