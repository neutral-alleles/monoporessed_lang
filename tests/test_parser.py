# tests/test_parser.py
import unittest
import logging
import sys

# Configure logging for tests to see parser output
# logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(levelname)s:%(message)s')

# Try importing modules first
import src.lexer
import src.mp_ast
import src.parser 

from src.lexer import Lexer, TT_EOF, TT_NEWLINE, TT_INDENT, TT_DEDENT, IndentationError
from src.parser import Parser, ParseError
from src.mp_ast import (Program, VarDecl, Assignment, IntegerLiteral,
                   BooleanLiteral, VariableRef, FloatLiteral, RuneLiteral, StringLiteral, 
                   IfStmt, WhileStmt, ForStmt, BinaryOp, ReturnStmt, UnaryOp,
                   IncrementStmt, DecrementStmt, SelfSetStmt)

class TestParser(unittest.TestCase):

    def assert_parses_to(self, code, expected_program_repr):
        """Asserts that the code parses to the expected AST repr."""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        # print("\nTokens:", tokens) # Debugging
        parser = Parser(tokens)
        try:
            program = parser.parse()
            # print("\nParsed AST:", repr(program)) # Debugging
            # print("Expected AST:", expected_program_repr)
            self.assertEqual(repr(program), expected_program_repr)
        except (ParseError, IndentationError) as e:
            # print("\nParse Error:", e) # Debugging
            self.fail(f"Parsing failed with error: {e}\nCode:\n{code}")
        except Exception as e:
            # print("\nUnexpected Error:", e) # Debugging
            self.fail(f"An unexpected error occurred during parsing: {e}\nCode:\n{code}")


    def assert_parse_error(self, code, expected_error_part):
        """Helper method to check for expected ParseError."""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        with self.assertRaises(ParseError) as cm:
            parser.parse()
        self.assertIn(expected_error_part, str(cm.exception))

    # --- Valid Tests ---

    def test_simple_declaration(self):
        code = "declare int32 myVar"
        expected = "Program([VarDecl(int32 myVar @1:1)])"
        self.assert_parses_to(code, expected)

    def test_declaration_with_newlines(self):
        code = "\n\ndeclare\nstring\nmessage\n\n"
        expected = "Program([VarDecl(string message @3:1)])"
        self.assert_parses_to(code, expected)

    def test_simple_assignment_integer(self):
        code = "set count 123"
        expected = "Program([Assign(count = Int(123 @1:11) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_simple_assignment_boolean(self):
        code = "set is_done true"
        expected = "Program([Assign(is_done = Bool(True @1:13) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_simple_assignment_float(self):
        code = "set price 99.9"
        expected = "Program([Assign(price = Float(99.9 @1:11) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_simple_assignment_rune(self):
        code = "set initial 'a'"
        expected = "Program([Assign(initial = Rune('a' @1:13) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_simple_assignment_string(self):
        code = 'set message "Hello"'
        expected = 'Program([Assign(message = String("Hello" @1:13) @1:1)])'
        self.assert_parses_to(code, expected)

    def test_assignment_variable_ref(self):
        code = "set new_var old_var"
        expected = "Program([Assign(new_var = VarRef(old_var @1:13) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_multiple_statements(self):
        # Reverted: This code is syntactically invalid due to indent,
        # should fail parsing later.
        code = """
declare int32 x
    set x 1 # Invalid indent
declare int32 y
set y 2
"""
        # Expecting ParseError because the parser now errors on top-level INDENT
        self.assert_parse_error(code, "Unexpected INDENT at top level")

    def test_multiple_statements_valid(self):
        code = """
declare int32 x
set x 1
declare int32 y
set y 2
"""
        expected = "Program([VarDecl(int32 x @2:1), Assign(x = Int(1 @3:7) @3:1), VarDecl(int32 y @4:1), Assign(y = Int(2 @5:7) @5:1)])"
        self.assert_parses_to(code, expected)

    def test_empty_input(self):
        code = ""
        expected = "Program([])"
        self.assert_parses_to(code, expected)

    def test_only_newlines(self):
        code = "\n\n\n"
        expected = "Program([])"
        self.assert_parses_to(code, expected)
        
    def test_only_comments_and_newlines(self):
        # Comments (even indented ones) and newlines should parse to empty program.
        code = """

# Comment line 1

    # Indented comment line 2
# Comment line 3

"""
        expected = "Program([])"
        self.assert_parses_to(code, expected)
        # self.assert_parse_error(code, "Unexpected INDENT at top level") # Old expectation

    # --- Invalid Tests ---

    def test_invalid_declaration_missing_type(self):
        code = "declare myVar"
        self.assert_parse_error(code, "Expected type after 'declare'")

    def test_invalid_declaration_missing_identifier(self):
        code = "declare int32"
        self.assert_parse_error(code, "Expected identifier after type")

    def test_invalid_declaration_wrong_order(self):
        code = "declare myVar int32"
        self.assert_parse_error(code, "Expected type after 'declare'")

    def test_invalid_assignment_missing_identifier(self):
        code = "set 123"
        self.assert_parse_error(code, "Expected identifier after 'set'")

    def test_invalid_assignment_missing_expression(self):
        code = "set myVar"
        self.assert_parse_error(code, "Unexpected EOF while parsing primary expression")

    def test_invalid_assignment_extra_token(self):
        code = "set myVar 10 extra"
        # Changed back to assert_parse_error
        # Updated expected error to mention IDENTIFIER
        self.assert_parse_error(code, "Expected newline after top-level statement, found IDENTIFIER")

    def test_invalid_statement_start(self):
        code = "123 set x"
        # Updated expected error to be more specific
        self.assert_parse_error(code, "Unexpected token type \'INTEGER\' with value \'123\' at start of statement")

    def test_invalid_token_in_expression(self):
        code = "set x ,"
        self.assert_parse_error(code, "Unexpected token type 'COMMA' in primary expression")
        
    def test_eof_after_declare(self):
        code = "declare"
        self.assert_parse_error(code, "Expected type after 'declare'")
        
    def test_eof_after_declare_type(self):
        code = "declare int32"
        self.assert_parse_error(code, "Expected identifier after type")

    def test_eof_after_set(self):
        code = "set"
        self.assert_parse_error(code, "Expected identifier after 'set'")
        
    def test_eof_after_set_identifier(self):
        code = "set myVar"
        self.assert_parse_error(code, "Unexpected EOF while parsing primary expression")

    # --- New Arithmetic Tests --- 
    def test_simple_add(self):
        code = "set result add 10 5"
        # Pos: 1(set) 4( ) 5(res) 11( ) 12(add) 15( ) 16(10) 18( ) 19(5)
        # Expected: add @ 1:11, 10 @ 1:15, 5 @ 1:18
        expected = "Program([Assign(result = Op(add Int(10 @1:16) Int(5 @1:19) @1:12) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_simple_fsub(self):
        code = "set total fsub 100.0 0.5"
        # Pos: 1(set) 4( ) 5(tot) 10( ) 11(fsub) 15( ) 16(100.0) 21( ) 22(0.5)
        # Expected: fsub @ 1:10, 100.0 @ 1:15, 0.5 @ 1:21 -> Corrected expected cols
        expected = "Program([Assign(total = Op(fsub Float(100.0 @1:16) Float(0.5 @1:22) @1:11) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_nested_ops(self):
        code = "set val add 5 mul 2 3"
        # Expected: add @ 1:8, 5 @ 1:12, (mul @ 1:14, 2 @ 1:18, 3 @ 1:20)
        expected = "Program([Assign(val = Op(add Int(5 @1:13) Op(mul Int(2 @1:19) Int(3 @1:21) @1:15) @1:9) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_op_with_vars(self):
        code = "set a add b c"
        # Expected: add @ 1:6, b @ 1:10, c @ 1:12 -> Corrected expected cols
        expected = "Program([Assign(a = Op(add VarRef(b @1:11) VarRef(c @1:13) @1:7) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_invalid_op_missing_operand1(self):
        code = "set x add"
        self.assert_parse_error(code, "Unexpected EOF while parsing primary expression")
        
    def test_invalid_op_missing_operand2(self):
        code = "set x add 1"
        self.assert_parse_error(code, "Unexpected EOF while parsing primary expression")
        
    def test_invalid_op_extra_operand(self):
        code = "set x add 1 2 3"
        # Changed back to assert_parse_error
        self.assert_parse_error(code, "Expected newline after top-level statement, found INTEGER")

    # --- New Comparison Operator Tests ---
    def test_comparison_eq(self):
        code = "set result eq count 10"
        # Expected: eq @ 1:11, count @ 1:14, 10 @ 1:20
        expected = "Program([Assign(result = Op(eq VarRef(count @1:15) Int(10 @1:21) @1:12) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_comparison_lt_vars(self):
        code = "set less lt price limit"
        # Expected: lt @ 1:9, price @ 1:12, limit @ 1:18
        expected = "Program([Assign(less = Op(lt VarRef(price @1:13) VarRef(limit @1:19) @1:10) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_comparison_nested_right(self):
        code = "set x eq a lt b c"
        # Expected: eq @ 1:6, a @ 1:9, (lt @ 1:11, b @ 1:14, c @ 1:16)
        expected = "Program([Assign(x = Op(eq VarRef(a @1:10) Op(lt VarRef(b @1:15) VarRef(c @1:17) @1:12) @1:7) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_comparison_nested_left(self):
        code = "set x eq lt a b c"
        # Expected: eq @ 1:6, (lt @ 1:9, a @ 1:12, b @ 1:14), c @ 1:16
        expected = "Program([Assign(x = Op(eq Op(lt VarRef(a @1:13) VarRef(b @1:15) @1:10) VarRef(c @1:17) @1:7) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_invalid_comparison_missing_operand(self):
        code = "set res gt val"
        self.assert_parse_error(code, "Unexpected EOF while parsing primary expression")

    # --- New Logical Operator Tests ---
    def test_logical_not(self):
        code = "set result not is_valid"
        # Expected: not @ 1:11, is_valid @ 1:15
        expected = "Program([Assign(result = Op(not VarRef(is_valid @1:16) @1:12) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_logical_not_literal(self):
        code = "set flag not true"
        # Expected: not @ 1:9, true @ 1:13
        expected = "Program([Assign(flag = Op(not Bool(True @1:14) @1:10) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_logical_and(self):
        code = "set both and has_key has_perms"
        # Expected: and @ 1:9, has_key @ 1:13, has_perms @ 1:21
        expected = "Program([Assign(both = Op(and VarRef(has_key @1:14) VarRef(has_perms @1:22) @1:10) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_logical_or(self):
        code = "set either or is_admin is_owner"
        # Expected: or @ 1:11, is_admin @ 1:14, is_owner @ 1:23
        expected = "Program([Assign(either = Op(or VarRef(is_admin @1:15) VarRef(is_owner @1:24) @1:12) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_nested_logical(self):
        code = "set complex and a or b not c"
        # Expected: and @ 1:12, a @ 1:16, (or @ 1:18, b @ 1:21, (not @ 1:23, c @ 1:27))
        expected = "Program([Assign(complex = Op(and VarRef(a @1:17) Op(or VarRef(b @1:22) Op(not VarRef(c @1:28) @1:24) @1:19) @1:13) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_mixed_logical_comparison(self):
        code = "set complex and eq a 10 not lt b 5"
        # Expected: and @ 1:12, (eq @ 1:16, a @ 1:19, 10 @ 1:21), (not @ 1:24, (lt @ 1:28, b @ 1:31, 5 @ 1:33))
        expected = "Program([Assign(complex = Op(and Op(eq VarRef(a @1:20) Int(10 @1:22) @1:17) Op(not Op(lt VarRef(b @1:32) Int(5 @1:34) @1:29) @1:25) @1:13) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_invalid_not_missing_operand(self):
        code = "set x not"
        self.assert_parse_error(code, "Unexpected EOF while parsing primary expression")
        
    def test_invalid_and_missing_operands(self):
        code = "set x and a"
        self.assert_parse_error(code, "Unexpected EOF while parsing primary expression")

    # --- New Return Statement Tests --- 
    def test_return_value(self):
        code = "return 10"
        expected = "Program([Return(Int(10 @1:8) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_return_variable(self):
        code = "return my_var"
        expected = "Program([Return(VarRef(my_var @1:8) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_return_expression(self):
        code = "return add x 1"
        expected = "Program([Return(Op(add VarRef(x @1:12) Int(1 @1:14) @1:8) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_return_no_value(self):
        code = "return"
        expected = "Program([Return(None @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_return_followed_by_newline(self):
        code = "return\n"
        expected = "Program([Return(None @1:1)])" 
        self.assert_parses_to(code, expected)
        
    def test_invalid_return_extra_tokens(self):
        code = "return 10 20"
        # Changed back to assert_parse_error
        self.assert_parse_error(code, "Expected newline after top-level statement, found INTEGER")

    # --- New Control Flow Tests ---
    def test_if_statement_simple(self):
        code = """
if true
    set x 1 # Use 4 spaces
"""
        # Expected: if @ 2:1, true @ 2:3, set @ 3:5, x @ 3:9, 1 @ 3:10
        expected = "Program([If(Bool(True @2:4), then=[Assign(x = Int(1 @3:11) @3:5)], else=None @2:1)])"
        self.assert_parses_to(code, expected)
        
    def test_if_else_statement(self):
        code = """
if lt a 0
    set sign -1 # Use 4 spaces
else
    set sign 1  # Use 4 spaces
"""
        # Expected: if @ 2:1, lt @ 2:3, a @ 2:6, 0 @ 2:8 | set @ 3:5, sign @ 3:9, -1 @ 3:13 | else @ 4:1 | set @ 5:5, sign @ 5:9, 1 @ 5:13
        expected = "Program([If(Op(lt VarRef(a @2:7) Int(0 @2:9) @2:4), then=[Assign(sign = Int(-1 @3:14) @3:5)], else=[Assign(sign = Int(1 @5:14) @5:5)] @2:1)])"
        self.assert_parses_to(code, expected)

    def test_if_nested(self):
        code = """
if a
    if b      # 4 spaces
        set x 1 # 8 spaces
else
    set y 2     # 4 spaces
"""
        # Expected: if @ 2:1, a @ 2:3 | if @ 3:5, b @ 3:7 | set @ 4:9, x @ 4:13, 1 @ 4:14 | else @ 5:1 | set @ 6:5, y @ 6:9, 2 @ 6:10
        # Corrected: Removed extra pos for nested if (@3:7), corrected col for Int 2 (@6:10 -> @6:11)
        expected = "Program([If(VarRef(a @2:4), then=[If(VarRef(b @3:8), then=[Assign(x = Int(1 @4:15) @4:9)], else=None @3:5)], else=[Assign(y = Int(2 @6:11) @6:5)] @2:1)])"
        self.assert_parses_to(code, expected)

    def test_while_statement(self):
        code = """
while lt i 10
    set i add i 1           # 4 spaces
    set total add total i   # 4 spaces
"""
        # Expected: while @ 2:1, lt @ 2:6, i @ 2:9, 10 @ 2:11 | set @ 3:5, i @ 3:9, add @ 3:11, i @ 3:15, 1 @ 3:17 | set @ 4:5, total @ 4:9, add @ 4:13, total @ 4:17, i @ 4:23
        # Corrected cols again
        expected = "Program([While(Op(lt VarRef(i @2:10) Int(10 @2:12) @2:7), body=[Assign(i = Op(add VarRef(i @3:15) Int(1 @3:17) @3:11) @3:5), Assign(total = Op(add VarRef(total @4:19) VarRef(i @4:25) @4:15) @4:5)] @2:1)])"
        self.assert_parses_to(code, expected)

    def test_for_statement(self):
        code = """
for i in range 10
    set total add total i # 4 spaces
"""
        # Expected: for @ 2:1, i @ 2:5, in @ 2:7, range @ 2:10, 10 @ 2:15 | set @ 3:5, total @ 3:9, add @ 3:13, total @ 3:17, i @ 3:23
        # Corrected cols again
        expected = "Program([For(i in Int(10 @2:16), body=[Assign(total = Op(add VarRef(total @3:19) VarRef(i @3:25) @3:15) @3:5)] @2:1)])"
        self.assert_parses_to(code, expected)
        
    def test_invalid_if_no_newline(self):
        code = "if true set x 1"
        self.assert_parse_error(code, "Expected newline after 'if' condition")

    def test_invalid_if_no_indent(self):
        code = "if true\nset x 1"
        self.assert_parse_error(code, "Expected indented block")
        
    def test_invalid_else_no_newline(self):
        code = """
if false
    set x 1 # Use 4 spaces
else set y 2 # Missing newline, space used instead of tab
"""
        # Now expecting newline error after fixing the tab issue
        self.assert_parse_error(code, "Expected newline after 'else'")
        
    def test_invalid_for_bad_header(self):
        code = "for i range 10\n    set x i" # Using 4 spaces now
        # Updated error message slightly to match parser output
        # Corrected expected error message to be more precise
        self.assert_parse_error(code, "Expected 'in' keyword near token KEYWORD('range' @1:7)")

    def test_block_mismatched_dedent(self):
        code = """
if true
    set x 1
 set y 2 # Only 1 space dedent
"""
        # Expecting specific IndentationError from lexer
        lexer = Lexer(code)
        # Updated expectation to match the actual error from lexer validation
        with self.assertRaisesRegex(IndentationError, "Invalid indentation level"):
             lexer.tokenize()

    # --- Increment/Decrement/SelfSet Tests (Simplified Syntax) ---
    def test_increment_simple_default(self):
        # Test 'inc x' (defaults to inc by 1)
        code = "inc counter"
        # AST: IncrementStmt(var_name='counter', delta=None) -> repr: Inc(counter @1:1)
        expected = "Program([Inc(counter @1:1)])"
        self.assert_parses_to(code, expected)

    def test_decrement_simple_default(self):
        # Test 'dec x' (defaults to dec by 1)
        code = "dec fuel"
        # AST: DecrementStmt(var_name='fuel', delta=None) -> repr: Dec(fuel @1:1)
        expected = "Program([Dec(fuel @1:1)])"
        self.assert_parses_to(code, expected)

    def test_increment_with_value(self):
        # Test 'inc x <expr>'
        code = "inc score 10"
        # AST: IncrementStmt(var_name='score', delta=IntegerLiteral(10)) -> repr: Inc(score by Int(10 @1:11) @1:1)
        # Corrected col: 10 @1:12 -> @1:11 to match actual output
        expected = "Program([Inc(score by Int(10 @1:11) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_decrement_with_expr(self):
        code = "dec fuel sub capacity used"
        # Expected: dec @ 1:1, fuel @ 1:5 | sub @ 1:9, capacity @ 1:13, used @ 1:22
        expected = "Program([Dec(fuel by Op(sub VarRef(capacity @1:14) VarRef(used @1:23) @1:10) @1:1)])"
        self.assert_parses_to(code, expected)
        
    def test_selfset_simple(self):
        code = "selfset total item_cost"
        # Expected: selfset @ 1:1, total @ 1:9, item_cost @ 1:14
        expected = "Program([SelfSet(total += VarRef(item_cost @1:15) @1:1)])"
        self.assert_parses_to(code, expected)

    def test_selfset_complex_expr(self):
        code = "selfset factor sub 1 0.1"
        # Expected: selfset @ 1:1, factor @ 1:9 | sub @ 1:14, 1 @ 1:18, 0.1 @ 1:20
        # Corrected cols again
        expected = "Program([SelfSet(factor += Op(sub Int(1 @1:20) Float(0.1 @1:22) @1:16) @1:1)])"
        self.assert_parses_to(code, expected)

    # --- Invalid inc/dec/selfset tests (Simplified Syntax) ---

    def test_invalid_inc_no_identifier(self):
        code = "inc 10"
        self.assert_parse_error(code, "Expected identifier after 'inc'")

    def test_invalid_dec_no_identifier(self):
        code = "dec sub a b"
        self.assert_parse_error(code, "Expected identifier after 'dec'")

    # Check that providing invalid token *instead* of expression fails correctly
    def test_invalid_inc_bad_token_after_ident(self):
        code = "inc counter declare"
        self.assert_parse_error(code, "Unexpected token type 'KEYWORD' in primary expression")
        
    def test_invalid_dec_bad_token_after_ident(self):
        code = "dec fuel :"
        self.assert_parse_error(code, "Unexpected token type 'COLON' in primary expression")
        
    def test_invalid_selfset_no_identifier(self):
        code = "selfset + x 1"
        self.assert_parse_error(code, "Expected variable name after 'selfset'")

    def test_invalid_selfset_no_expression(self):
        code = "selfset myVar"
        self.assert_parse_error(code, "Unexpected EOF while parsing primary expression")

    # --- Combining with Control Flow (Simplified Syntax) ---

    def test_inc_in_loop(self):
        code = """
while true
    inc i
"""
        # Expected: while @ 2:1, true @ 2:6 | inc @ 3:5, i @ 3:9
        # Corrected: Inc repr for default delta only shows keyword pos
        expected = "Program([While(Bool(True @2:7), body=[Inc(i @3:5)] @2:1)])"
        self.assert_parses_to(code, expected)

    def test_dec_in_loop_with_value(self):
        code = """
for i in range 10
    dec counter 2
"""
        # Expected: for @ 2:1, i @ 2:5, in @ 2:7, range @ 2:10, 10 @ 2:15 | dec @ 3:5, counter @ 3:9, 2 @ 3:16
        expected = "Program([For(i in Int(10 @2:16), body=[Dec(counter by Int(2 @3:17) @3:5)] @2:1)])"
        self.assert_parses_to(code, expected)

    def test_selfset_in_if(self):
        code = """
if x
    selfset x 2 # Use 4 spaces
"""
        # Expected: if @ 2:1, x @ 2:3 | selfset @ 3:5, x @ 3:13, 2 @ 3:14
        expected = "Program([If(VarRef(x @2:4), then=[SelfSet(x += Int(2 @3:15) @3:5)], else=None @2:1)])"
        self.assert_parses_to(code, expected)

# Ensure main execution block is present and correctly indented
if __name__ == '__main__':
    unittest.main() 