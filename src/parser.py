# src/parser.py
import logging
from typing import List, Optional, Union, Any, Tuple

# Use relative imports assuming src is treated as a package
from .lexer import (Token, TT_BOOLEAN, TT_COLON, TT_COMMA, TT_EOF, TT_FLOAT,
                    TT_IDENTIFIER, TT_INTEGER, TT_INVALID, TT_KEYWORD, KEYWORDS,
                    TT_LPAREN, TT_RPAREN, # Added for function parsing
                    TT_NEWLINE, TT_INDENT, TT_DEDENT, # Added INDENT/DEDENT
                    TT_RUNE, TT_STRING, TT_TYPE, IndentationError) # Import IndentationError
from .mp_ast import (ASTNode, Program, Statement, VarDef, Assignment, Expression,
                   IntegerLiteral, FloatLiteral, BooleanLiteral, RuneLiteral,
                   StringLiteral, VariableRef, BinaryOp, ReturnStmt, UnaryOp,
                   IfStmt, WhileStmt, ForStmt,
                   IncrementStmt, DecrementStmt, SelfSetStmt,
                   FuncDef, FuncParam, FuncCall, ReadIntExpr, # Removed PrintStmt import
                   BreakStmt, ContinueStmt) # <<< Added BreakStmt, ContinueStmt

# Setup logger
log = logging.getLogger('parser')
# Ensure logger propagates to root logger configured elsewhere
# Basic config if run standalone for debugging
if not log.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(levelname)s:%(message)s')


class ParseError(Exception):
    def __init__(self, message: str, token: Optional[Token] = None):
        if token:
            super().__init__(f"{message} near token {token} ({token.line}:{token.col})")
        else:
            super().__init__(message)
        self.token = token

# --- Define prefix operator keywords ---
PREFIX_BINARY_KEYWORDS = {
    # Arithmetic
    'add', 'sub', 'mul', 'div',
    'fadd', 'fsub', 'fmul', 'fdiv',
    # Comparison
    'eq', 'neq', 'lt', 'lte', 'gt', 'gte',
    # Logical (keeping as keywords for now)
    'and', 'or',
    # Bitwise (placeholder)
    'xor', 
}
PREFIX_UNARY_KEYWORDS = {
    'not'
}
# Keep track of which keywords expect specific numbers of arguments for parsing
# This is a SIMPLIFICATION for now, assuming all binary ops take 2, unary take 1.
# Does not account for built-ins like print_int which look like unary ops syntactically.
OPERATOR_ARITY = {kw: 2 for kw in PREFIX_BINARY_KEYWORDS}
OPERATOR_ARITY.update({kw: 1 for kw in PREFIX_UNARY_KEYWORDS})

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens #[t for t in tokens if t.type not in (TT_COMMENT, TT_TAB)]
        self.current_token_index = 0
        self.current_token = self.tokens[self.current_token_index] if self.tokens else None
        self.previous_token = None
        self.logger = logging.getLogger(__name__)
        # Basic config - can be adjusted later
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG) # Or INFO, WARNING, etc.
        self._log_state("Parser initialized")

    def _log_state(self, message: str):
        """Log the current state of the parser."""
        token_info = f"Token={self.current_token}" if self.current_token else "Token=None (EOF)"
        log.debug(f"[Parser State @ Pos {self.current_token_index}] {message}. {token_info}")

    def _log_entry(self, func_name: str):
        """Log entry into a parsing function."""
        log.debug(f"--> Entering {func_name}")
        self._log_state("Start state")

    def _log_exit(self, func_name: str, result: Optional[Any]):
        """Log exit from a parsing function."""
        result_repr = repr(result) if result is not None else "None"
        # Limit result repr length for logs
        if len(result_repr) > 100:
            result_repr = result_repr[:97] + "..."
        log.debug(f"<-- Exiting {func_name}. Result: {result_repr}")
        self._log_state("End state")

    def _advance(self):
        """Advance to the next token."""
        self.current_token_index += 1
        if self.current_token_index < len(self.tokens):
            self.current_token = self.tokens[self.current_token_index]
            # self._log_state("Advanced") # Can be too noisy
        else:
            self.current_token = None # Signifies EOF
            # self._log_state("Advanced to EOF")

    def _peek(self, offset: int = 1) -> Optional[Token]:
        """Look ahead at a future token without advancing."""
        peek_pos = self.current_token_index + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None

    def _consume(self, token_type: str, message: Optional[str] = None) -> Token:
        """Consume the current token if it matches the type, otherwise raise error."""
        token = self.current_token
        log.debug(f"Attempting to consume {token_type}. Current: {token}")
        if token and token.type == token_type:
            self._advance()
            log.debug(f"Successfully consumed {token_type}.")
            return token
        else:
            expected_msg = f"Expected {token_type}"
            error_msg = f"{message} ({expected_msg})" if message else expected_msg
            raise ParseError(error_msg, token)

    def _check(self, token_type: str, value: Optional[str] = None) -> bool:
        """Check if the current token matches the given type and optionally value."""
        if not self.current_token:
            return False
        if self.current_token.type == token_type:
            if value is None:
                return True
            # Handle keyword checking specifically
            if token_type == TT_KEYWORD and self.current_token.value == value:
                return True
            # Could extend for other types with specific values if needed
        return False

    def _skip_newlines(self):
        """Consume one or more consecutive NEWLINE tokens."""
        skipped = False
        while self.current_token and self.current_token.type == TT_NEWLINE:
            log.debug("Skipping NEWLINE")
            self._advance()
            skipped = True
        return skipped

    def parse(self) -> Program:
        """Top-level parsing method."""
        self._log_entry("parse")
        program = self.parse_program()
        if self.current_token and self.current_token.type != TT_EOF:
             log.warning(f"Parser finished but did not reach EOF. Remaining token: {self.current_token}")
             # Depending on strictness, could raise error here
        self._log_exit("parse", program)
        return program

    def parse_program(self) -> Program:
        self.logger.info("Parsing program")
        definitions = [] # Can be statements or function defs
        start_token = self.current_token  # Keep track for the Program node

        while not self._check(TT_EOF):
            # Skip leading newlines at the top level or between statements
            while self._check(TT_NEWLINE):
                self._advance()
            
            # If we reached EOF after skipping newlines, break
            if self._check(TT_EOF):
                break

            # Check for unexpected indentation *before* trying to parse a statement
            # This prevents errors on valid indented comments following blank lines.
            # If it's not EOF or NEWLINE, it *should* be the start of a statement.
            # Any INDENT/DEDENT here is an error.
            if self._check(TT_INDENT) or self._check(TT_DEDENT):
                 # Allow skipping potentially spurious indents/dedents if the line is empty/comment
                 # We need a way to peek ahead or check the line content.
                 # For now, raise error, revisit if needed.
                 # TODO: Check if skipping indent/dedent on comment/empty lines is better
                 raise ParseError(f"Unexpected {self.current_token.type} at top level", self.current_token)


            # Parse a single top-level definition (statement or function)
            try:
                definition = self.parse_definition() # Renamed from parse_statement
                if definition: # parse_definition might return None
                    definitions.append(definition)
                    
                    # Consume any newlines immediately following the definition
                    self._skip_newlines()

                    # After a valid definition and skipping newlines, we should either be at EOF 
                    # or the start of the next definition. Check if it's *not* EOF and *not* a valid start.
                    # We don't need a complex check here yet, because parse_definition handles invalid starts.
                    # The original check was too strict, causing the error. We simply proceed.
                    # If the next token is invalid, the next iteration's parse_definition will raise an error.

                # If parse_definition returned None (e.g., comment line), loop continues
            except ParseError as e:
                 self.logger.error(f"Error parsing definition: {e}")
                 raise # Re-raise the error to stop parsing

            # Safety break to prevent infinite loops if stuck
            # This shouldn't be strictly necessary if parsing advances correctly
            # Consider adding a check against current_token position change if needed

        program_node = Program(definitions)
        # Set position based on the span from the first token to the last
        if definitions and self.previous_token:
             program_node.set_pos(start_token.line, start_token.col, self.previous_token.end_line, self.previous_token.end_col)
        elif start_token: # Handle empty program case
             program_node.set_pos(start_token.line, start_token.col, start_token.line, start_token.col)

        self.logger.info(f"Parsed program: {program_node}")
        return program_node

    def parse_definition(self) -> Union[Statement, FuncDef, None]:
        self.logger.debug(f"Parsing definition/statement at {self.current_token}")

        # Skip comments and blank lines *before* deciding what statement it is
        while self._check(TT_NEWLINE): # or self._check(TT_COMMENT): # Lexer skips comments now
             self._advance()

        if self._check(TT_EOF):
            self.logger.debug("EOF reached, returning None for definition")
            return None # Return None if only newlines/comments found until EOF


        token = self.current_token
        self.logger.debug(f"Current token for definition parsing: {token}")

        # --- Check for different definition/statement types --- 
        if self._check(TT_KEYWORD, 'def'):      
            return self.parse_function_definition()
        elif self._check(TT_KEYWORD, 'let') or self._check(TT_KEYWORD, 'init'):
            return self._parse_var_def()
        elif self._check(TT_KEYWORD, 'set'):
            return self.parse_assignment()
        elif self._check(TT_KEYWORD, 'inc'):
            return self.parse_increment_statement()
        elif self._check(TT_KEYWORD, 'dec'):
            return self.parse_decrement_statement()
        elif self._check(TT_KEYWORD, 'selfset'):
            return self.parse_selfset_statement()
        elif self._check(TT_KEYWORD, 'return'):
            return self.parse_return_statement()
        elif self._check(TT_KEYWORD, 'if'):
            return self.parse_if_statement()
        elif self._check(TT_KEYWORD, 'while'):
            return self.parse_while_statement()
        elif self._check(TT_KEYWORD, 'for'):
            return self.parse_for_statement()
        elif self._check(TT_KEYWORD, 'break'):
            return self._parse_break_statement()
        elif self._check(TT_KEYWORD, 'continue'):
            return self._parse_continue_statement()
        elif self._check(TT_IDENTIFIER):
            # If it starts with an identifier, try parsing it as an expression.
            # Our parse_expression logic will handle turning 'func arg' into FuncCall.
            # If it's just 'var', parse_expression will return VariableRef, which isn't 
            # a valid standalone statement, but we might allow this later or handle errors.
            # For now, assume an identifier start means a potential function call statement.
            log.debug("Identifier found at start of definition/statement, attempting to parse as expression/FuncCall.")
            # We expect the result of parse_expression to be added to the program directly
            # if it's a valid top-level construct (like FuncCall based on updated Program hint)
            expr = self.parse_expression()
            # Here, we might want to validate that expr is actually a FuncCall if we ONLY want calls as statements
            # For now, let's return the expression directly.
            return expr # Return the parsed expression (FuncCall or potentially others)
        # If it's none of the keywords, it might be an expression statement (if allowed)
        # or an error. Our language doesn't have standalone expression statements at top level.
        else:
             # Check again for EOF or newline in case we skipped comments/newlines initially
             if self._check(TT_EOF) or self._check(TT_NEWLINE):
                 self.logger.debug("Found only comments/newlines, returning None")
                 return None # Effectively skipped comment/empty lines
             # If it's not a recognized keyword, EOF, or NEWLINE, it's an error.
             raise ParseError(f"Unexpected token type '{token.type}' with value '{token.value}' at start of definition/statement", token)

    # --- Block Parsing ---
    def parse_block(self) -> List[Statement]:
        """Parses an indented block of statements."""
        self._log_entry("parse_block")
        statements: List[Statement] = []
        self._consume(TT_INDENT, "Expected indented block")

        while True: # Loop indefinitely until DEDENT or EOF
            # Skip newlines at the start of a line within the block
            while self._check(TT_NEWLINE):
                self._advance()

            # Check for the end of the block *after* skipping newlines
            if not self.current_token or self._check(TT_DEDENT):
                break # Exit loop if DEDENT or EOF is encountered

            log.debug("Parsing next statement in block...")
            try:
                statement = self.parse_definition()
                if statement:
                    statements.append(statement)
                # If parse_definition returns None (comment/blank line), loop continues naturally
            except ParseError as e:
                # Re-raise errors encountered during statement parsing within the block
                self.logger.error(f"Error parsing statement within block: {e}")
                raise e

        # Consume the closing DEDENT token
        self._consume(TT_DEDENT, "Expected DEDENT to close indented block")
        self._log_exit("parse_block", statements)
        return statements
    # --- End Block Parsing ---

    # --- Control Flow Statement Parsing ---
    def parse_if_statement(self) -> IfStmt:
        """Parses 'if <condition> NEWLINE INDENT <statements> DEDENT [else NEWLINE INDENT <statements> DEDENT]'."""
        self._log_entry("parse_if_statement")
        start_token = self._consume(TT_KEYWORD, "Expected 'if'")
        if start_token.value != 'if': raise ParseError("Expected 'if'", start_token)

        condition = self.parse_expression()
        self._consume(TT_NEWLINE, "Expected newline after 'if' condition")

        log.debug("Parsing 'if' block")
        if_body = self.parse_block()

        elif_clauses: Optional[List[Tuple[Expression, List[Statement]]]] = []
        # Check for 'elif' right after the if-block's DEDENT
        while self.current_token and self.current_token.type == TT_KEYWORD and self.current_token.value == "elif":
            log.debug("Found 'elif' keyword")
            elif_token = self._advance() # Consume 'elif'
            
            elif_condition = self.parse_expression()
            self._consume(TT_NEWLINE, "Expected newline after 'elif' condition")
            
            log.debug("Parsing 'elif' block")
            elif_body = self.parse_block()
            
            if elif_clauses is None: elif_clauses = [] # Initialize list on first elif
            elif_clauses.append((elif_condition, elif_body))

        else_body: Optional[List[Statement]] = None
        # Check for 'else' right after the last if/elif block's DEDENT
        if self.current_token and self.current_token.type == TT_KEYWORD and self.current_token.value == "else":
            log.debug("Found 'else' keyword")
            self._advance() # Consume 'else'
            self._consume(TT_NEWLINE, "Expected newline after 'else'")
            log.debug("Parsing 'else' block")
            else_body = self.parse_block()

        # If elif_clauses remained empty, set it back to None for the AST node
        if not elif_clauses: 
            elif_clauses = None 
            
        stmt = IfStmt(condition=condition, if_body=if_body, 
                      elif_clauses=elif_clauses, # Pass the collected elifs
                      else_body=else_body,
                      line=start_token.line, col=start_token.col)
        self._log_exit("parse_if_statement", stmt)
        return stmt

    def parse_while_statement(self) -> WhileStmt:
        """Parses 'while <condition> NEWLINE INDENT <statements> DEDENT'."""
        self._log_entry("parse_while_statement")
        start_token = self._consume(TT_KEYWORD, "Expected 'while'")
        if start_token.value != 'while': raise ParseError("Expected 'while'", start_token)

        condition = self.parse_expression()
        self._consume(TT_NEWLINE, "Expected newline after 'while' condition")

        log.debug("Parsing 'while' block")
        body = self.parse_block()

        stmt = WhileStmt(condition=condition, body=body, line=start_token.line, col=start_token.col)
        self._log_exit("parse_while_statement", stmt)
        return stmt

    def parse_for_statement(self) -> ForStmt:
        # Simple: for <var> in range <N>
        """Parses 'for <var> in range <expr> NEWLINE INDENT <statements> DEDENT'."""
        self._log_entry("parse_for_statement")
        start_token = self._consume(TT_KEYWORD, "Expected 'for'")
        if start_token.value != 'for': raise ParseError("Expected 'for'", start_token)

        iterator_token = self._consume(TT_IDENTIFIER, "Expected iterator variable after 'for'")

        in_token = self._consume(TT_KEYWORD, "Expected 'in' after iterator variable")
        if in_token.value != 'in': raise ParseError("Expected 'in' keyword", in_token)

        range_token = self._consume(TT_KEYWORD, "Expected 'range' after 'in'")
        if range_token.value != 'range': raise ParseError("Expected 'range' keyword", range_token)

        iterable_expr = self.parse_expression() # Expecting integer N
        # TODO: Validate iterable_expr is integer or add range object?

        self._consume(TT_NEWLINE, "Expected newline after 'for' header")

        log.debug("Parsing 'for' block")
        body = self.parse_block()

        stmt = ForStmt(iterator_var=iterator_token.value, iterable=iterable_expr, body=body,
                       line=start_token.line, col=start_token.col)
        self._log_exit("parse_for_statement", stmt)
        return stmt
    # --- End Control Flow ---

    # --- Increment/Decrement Statement Parsing ---
    def parse_increment_statement(self) -> IncrementStmt: # Changed back to non-optional
        """Parses 'inc <identifier> [<expression>]'. Defaults to inc by 1 if no expression."""
        self._log_entry("parse_increment_statement")
        start_token = self._consume(TT_KEYWORD, "Expected 'inc'")
        if start_token.value != 'inc': raise ParseError("Internal error: Expected 'inc' keyword", start_token)

        # Expect identifier
        # Use consume directly, which raises ParseError on failure
        ident_token = self._consume(TT_IDENTIFIER, "Expected identifier after 'inc'")

        delta: Optional[Expression] = None
        # Check if an expression follows directly (not newline, EOF, dedent)
        if self.current_token and self.current_token.type not in [TT_NEWLINE, TT_EOF, TT_DEDENT]:
            log.debug(f"Parsing optional delta expression for inc {ident_token.value}")
            # parse_expression now raises error on failure, no need to check for None
            delta = self.parse_expression() 
        else:
            log.debug(f"No delta expression found for inc {ident_token.value}, defaulting to 1.")
            # Delta remains None

        inc_stmt = IncrementStmt(var_name=ident_token.value, delta=delta, line=start_token.line, col=start_token.col)
        self._log_exit("parse_increment_statement", inc_stmt)
        return inc_stmt

    def parse_decrement_statement(self) -> DecrementStmt: # Changed back to non-optional
        """Parses 'dec <identifier> [<expression>]'. Defaults to dec by 1 if no expression."""
        self._log_entry("parse_decrement_statement")
        start_token = self._consume(TT_KEYWORD, "Expected 'dec'")
        if start_token.value != 'dec': raise ParseError("Internal error: Expected 'dec' keyword", start_token)

        # Expect identifier
        ident_token = self._consume(TT_IDENTIFIER, "Expected identifier after 'dec'")

        delta: Optional[Expression] = None
        # Check if an expression follows directly
        if self.current_token and self.current_token.type not in [TT_NEWLINE, TT_EOF, TT_DEDENT]:
            log.debug(f"Parsing optional delta expression for dec {ident_token.value}")
            # parse_expression now raises error on failure
            delta = self.parse_expression()
        else:
            log.debug(f"No delta expression found for dec {ident_token.value}, defaulting to 1.")
            # Delta remains None

        dec_stmt = DecrementStmt(var_name=ident_token.value, delta=delta, line=start_token.line, col=start_token.col)
        self._log_exit("parse_decrement_statement", dec_stmt)
        return dec_stmt

    def parse_assignment(self) -> Assignment: # Changed back to non-optional
        """Parses 'set <identifier> <expression>'."""
        self._log_entry("parse_assignment")
        start_token = self._consume(TT_KEYWORD, "Expected 'set'")
        if start_token.value != 'set': raise ParseError("Internal error: Expected 'set' keyword", start_token)

        # Expect identifier
        ident_token = self._consume(TT_IDENTIFIER, "Expected identifier after 'set'")

        # Expect expression IMMEDIATELY after identifier
        # parse_expression will raise error if it fails
        expression = self.parse_expression()

        assign = Assignment(var_name=ident_token.value, expression=expression, line=start_token.line, col=start_token.col)
        self._log_exit("parse_assignment", assign)
        return assign

    def parse_return_statement(self) -> ReturnStmt: # Changed back to non-optional
        """Parses 'return [<expression>]'."""
        self._log_entry("parse_return_statement")
        start_token = self._consume(TT_KEYWORD, "Expected 'return'")
        if start_token.value != 'return': raise ParseError("Internal error: Expected 'return' keyword", start_token)

        expression: Optional[Expression] = None
        # Check if there's an expression to return (anything not structural)
        if self.current_token and self.current_token.type not in [TT_NEWLINE, TT_EOF, TT_DEDENT]:
             # parse_expression raises error if it fails
            expression = self.parse_expression()

        ret_stmt = ReturnStmt(expression=expression, line=start_token.line, col=start_token.col)
        self._log_exit("parse_return_statement", ret_stmt)
        return ret_stmt

    def parse_selfset_statement(self) -> SelfSetStmt: # Changed back to non-optional
        """Parses 'selfset <identifier> <expression>'. Implicitly adds."""
        self._log_entry("parse_selfset_statement")
        start_token = self._consume(TT_KEYWORD, "Expected 'selfset'")
        if start_token.value != 'selfset': raise ParseError("Internal error: Expected 'selfset' keyword", start_token)

        # Expect identifier
        var_name_token = self._consume(TT_IDENTIFIER, "Expected variable name after 'selfset'")

        # Expect expression IMMEDIATELY after identifier
        # parse_expression raises error if it fails
        expression = self.parse_expression()

        selfset_stmt = SelfSetStmt(
            var_name=var_name_token.value,
            expression=expression,
            line=start_token.line,
            col=start_token.col
        )
        self._log_exit("parse_selfset_statement", selfset_stmt)
        return selfset_stmt

    # --- Expression Parsing (Prefix Ops, Calls, Atoms v5) ---

    def parse_expression(self) -> Expression:
        """Parses expression. Handles prefix ops, calls, atoms.
           Requires complex operands/args to be parenthesized (handled in _parse_atom).
        """
        self._log_entry("parse_expression")
        token = self.current_token

        if not token:
            raise ParseError("Unexpected EOF while parsing expression")

        expr = None
        # Check for Prefix Unary Operators
        if self._check(TT_KEYWORD) and token.value in PREFIX_UNARY_KEYWORDS:
            op_token = token
            op_value = token.value
            self._advance()
            log.debug(f"Parsing operand for unary op '{op_value}'")
            operand = self._parse_atom() # Operand must be an atom
            expr = UnaryOp(operator=op_value, operand=operand, line=op_token.line, col=op_token.col)
        
        # Check for Prefix Binary Operators
        elif self._check(TT_KEYWORD) and token.value in PREFIX_BINARY_KEYWORDS:
            op_token = token
            op_value = token.value
            self._advance()
            log.debug(f"Parsing operands for binary op '{op_value}'")
            operand1 = self._parse_atom() # Operand 1 must be an atom
            operand2 = self._parse_atom() # Operand 2 must be an atom
            expr = BinaryOp(operator=op_value, left=operand1, right=operand2, line=op_token.line, col=op_token.col)

        # Check for Identifier (potential function call or variable)
        elif token.type == TT_IDENTIFIER:
            identifier_token = token
            func_expr = VariableRef(identifier_token.value, line=identifier_token.line, col=identifier_token.col)
            self._advance() # Consume identifier
            log.debug(f"Parsed identifier '{identifier_token.value}', checking for arguments...")
            
            # Greedily parse arguments as atoms
            args = []
            while self._can_start_atom(self.current_token):
                log.debug(f"Parsing potential argument starting with {self.current_token}")
                args.append(self._parse_atom())
            
            if args: # If arguments were found, it's a function call
                log.debug(f"Found {len(args)} arguments. Treating '{identifier_token.value}' as function call.")
                expr = FuncCall(func_expr=func_expr, arguments=args, line=identifier_token.line, col=identifier_token.col)
            else: # No arguments found, it's just a variable reference
                log.debug(f"No arguments found. Treating '{identifier_token.value}' as variable reference.")
                expr = func_expr # Use the VariableRef created earlier
        
        # Check for Atoms (Literals, Parenthesized) - if none of the above matched
        else:
            log.debug(f"Token {token} not an operator or identifier, parsing as atom.")
            # This handles literals and parenthesized expressions directly
            expr = self._parse_atom() 

        # If expr is still None here, something went wrong
        if expr is None: 
             raise ParseError(f"Internal parser error: Could not parse expression starting with {token}", token)

        self._log_exit("parse_expression", expr)
        return expr

    def _parse_atom(self) -> Expression:
        """Parses atomic expressions: Literals, Variables (Identifiers), Parenthesized.
           It does NOT handle operators or function calls with arguments directly.
        """ 
        self._log_entry("_parse_atom")
        token = self.current_token
        
        if not token:
            raise ParseError("Unexpected EOF while parsing atomic expression")

        atom = None
        if token.type == TT_INTEGER:
            atom = IntegerLiteral(token.value, line=token.line, col=token.col)
            self._advance()
        elif token.type == TT_FLOAT:
            atom = FloatLiteral(token.value, line=token.line, col=token.col)
            self._advance()
        elif token.type == TT_BOOLEAN:
            atom = BooleanLiteral(token.value, line=token.line, col=token.col)
            self._advance()
        elif token.type == TT_RUNE:
            atom = RuneLiteral(token.value, line=token.line, col=token.col)
            self._advance()
        elif token.type == TT_STRING:
            atom = StringLiteral(token.value, line=token.line, col=token.col)
            self._advance()
        elif token.type == TT_IDENTIFIER: # Treat identifiers as variables here
            atom = VariableRef(token.value, line=token.line, col=token.col)
            self._advance()
        elif token.type == TT_LPAREN: # Handle parenthesized expressions
            paren_line, paren_col = token.line, token.col
            self._advance() # Consume '('
            log.debug("Parsing parenthesized expression within atom...")
            atom = self.parse_expression() # Recursive call for the content
            self._consume(TT_RPAREN, "Expected ')' after parenthesized expression")
            log.debug("Parsed parenthesized expression within atom.")
        else:
            raise ParseError(f"Unexpected token when parsing atomic expression: {token}", token)
        
        self._log_exit("_parse_atom", atom)
        return atom

    def _can_start_expression(self, token: Optional[Token]) -> bool:
        """Checks if a token can be the start of an expression (operand/atom/call/prefix-op)."""
        if not token:
            return False
        return token.type in (
            TT_INTEGER, TT_FLOAT, TT_BOOLEAN, TT_RUNE, TT_STRING, # Literals
            TT_IDENTIFIER, # Variable or Function call start
            TT_LPAREN      # Parenthesized expression
        ) or (
            token.type == TT_KEYWORD and 
            (token.value in PREFIX_UNARY_KEYWORDS or token.value in PREFIX_BINARY_KEYWORDS)
        )

    def _can_start_atom(self, token: Optional[Token]) -> bool:
        """Checks if a token can start an atomic expression (literal, identifier, '(')."""
        if not token:
            return False
        # An atom can be a literal, an identifier, or start with '('
        return token.type in (TT_INTEGER, TT_FLOAT, TT_BOOLEAN, TT_RUNE, TT_STRING,
                              TT_IDENTIFIER, TT_LPAREN)

    # --- Function Definition Parsing ---

    def parse_function_definition(self) -> FuncDef:
        """Parses 'def <name>(<params>) <return_type> NEWLINE INDENT <body> DEDENT'."""
        self._log_entry("parse_function_definition")
        start_token = self._consume(TT_KEYWORD, "Expected 'def'")
        if start_token.value != 'def': raise ParseError("Expected 'def'", start_token)

        func_name_token = self._consume(TT_IDENTIFIER, "Expected function name after 'def'")
        func_name = func_name_token.value

        self._consume(TT_LPAREN, "Expected '(' after function name")
        
        params: List[FuncParam] = []
        # Parse parameters until ')'
        if not self._check(TT_RPAREN):
            while True:
                param_type_token = self._consume(TT_TYPE, "Expected parameter type")
                param_name_token = self._consume(TT_IDENTIFIER, "Expected parameter name")
                params.append(FuncParam(name=param_name_token.value,
                                        type_name=param_type_token.value,
                                        line=param_type_token.line,
                                        col=param_type_token.col))
                
                if self._check(TT_RPAREN):
                    break
                self._consume(TT_COMMA, "Expected ',' or ')' after parameter")
        
        self._consume(TT_RPAREN, "Expected ')' after function parameters")
        
        # Parse return type
        return_type_token = self._consume(TT_TYPE, "Expected return type after parameters")
        return_type = return_type_token.value

        self._consume(TT_NEWLINE, "Expected newline after function signature")

        log.debug(f"Parsing function body for '{func_name}'")
        body = self.parse_block() # Use existing block parser

        func_def = FuncDef(name=func_name,
                           params=params,
                           return_type=return_type,
                           body=body,
                           line=start_token.line,
                           col=start_token.col)
        self._log_exit("parse_function_definition", func_def)
        return func_def

    # --- Add _parse_var_def ---
    def _parse_var_def(self) -> VarDef:
        """Parses 'init|let <type> <name> <expression>'."""
        self._log_entry("_parse_var_def")
        # Consume KEYWORD and check its value
        start_token = self._consume(TT_KEYWORD, "Expected 'init' or 'let' keyword")
        is_mutable = False
        if start_token.value == 'init':
            is_mutable = True
        elif start_token.value == 'let':
            is_mutable = False
        else:
            raise ParseError(f"Expected keyword 'init' or 'let', but got '{start_token.value}'", start_token)
        
        type_token = self._consume(TT_TYPE, f"Expected type after '{start_token.value}'")
        ident_token = self._consume(TT_IDENTIFIER, f"Expected identifier after type '{type_token.value}'")
        
        # Now parse the initialization expression
        expression = self.parse_expression()
        
        # Create VarDef node
        stmt = VarDef(
            is_mutable=is_mutable,
            var_type=type_token.value,
            var_name=ident_token.value,
            initial_expression=expression,
            line=start_token.line,
            col=start_token.col
        )
        self._log_exit("_parse_var_def", stmt)
        return stmt
    # --- End Function Definition --- 

    # --- Add Break/Continue Statement Parsing ---
    def _parse_break_statement(self) -> BreakStmt:
        """Parses 'break'."""
        self._log_entry("_parse_break_statement")
        token = self._consume(TT_KEYWORD, "Expected 'break' keyword")
        if token.value != 'break':
            raise ParseError(f"Expected keyword 'break', but got '{token.value}'", token)
        stmt = BreakStmt(line=token.line, col=token.col)
        self._log_exit("_parse_break_statement", stmt)
        return stmt

    def _parse_continue_statement(self) -> ContinueStmt:
        """Parses 'continue'."""
        self._log_entry("_parse_continue_statement")
        token = self._consume(TT_KEYWORD, "Expected 'continue' keyword")
        if token.value != 'continue':
            raise ParseError(f"Expected keyword 'continue', but got '{token.value}'", token)
        stmt = ContinueStmt(line=token.line, col=token.col)
        self._log_exit("_parse_continue_statement", stmt)
        return stmt
    # --- End Break/Continue ---