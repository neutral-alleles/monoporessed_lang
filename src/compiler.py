import sys
import os # Import os for file checks
from .lexer import Lexer, Token, TT_INVALID, IndentationError # Relative import
from .parser import Parser, ParseError # Relative import
from . import mp_ast # Relative import
from .mp_ast import ASTNode # Import base class for type hinting
# --- Add Semantic Analyzer imports ---
from .semantic_analyzer import SemanticAnalyzer, SemanticError
from .semantic_analyzer import SymbolTable
import logging

# --- Logger Setup ---
log = logging.getLogger('compiler')
# Basic config - assuming root logger is configured elsewhere or by runner script
# If run standalone, basicConfig might be needed in main

# Monopressed Language Compiler

# --- C Type Mapping ---
# Basic mapping from Monopressed types to C types
# Assumes <stdint.h> and <stdbool.h> will be included
C_TYPE_MAP = {
    'int8': 'int8_t',
    'int16': 'int16_t',
    'int32': 'int32_t',
    'int64': 'int64_t',
    'uint8': 'uint8_t',
    'uint16': 'uint16_t',
    'uint32': 'uint32_t',
    'uint64': 'uint64_t',
    'float32': 'float',
    'float64': 'double',
    'bool': 'bool',
    'rune': 'char', # Assuming rune maps to char for simplicity
    'string': 'char*', # Basic string mapping (needs memory management)
    'unit': 'void' # For return types
}

class CompilerError(Exception):
    pass

class Compiler:
    def __init__(self):
        self.c_code_lines = []
        self.indent_level = 0
        self.logger = logging.getLogger(__name__) # Use class/module specific logger

    def _add_line(self, line: str):
        indent = "    " * self.indent_level
        self.c_code_lines.append(indent + line)

    def _visit(self, node: ASTNode) -> str:
        """Dispatch to the appropriate visit method and return generated C snippet."""
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        self.logger.debug(f"Visiting {type(node).__name__} node using {visitor.__name__}")
        # Visitors should return the C code representation of the node
        # For statements, this might be the statement itself (e.g., "x = 5;")
        # For expressions, it's the expression value (e.g., "5", "x", "add(a, b)")
        result = visitor(node)
        if isinstance(result, str):
            # Log returned C code snippet for expression visitors
            self.logger.debug(f"  -> Generated C (expression): {result}")
        # Statement visitors log lines internally
        return result

    def generic_visit(self, node: ASTNode):
        # Ensure node is not None before trying to access its type
        if node is None:
            raise CompilerError("Attempted to visit a None node")
        raise CompilerError(f"No visit_{type(node).__name__} method defined for node {type(node)}")

    def compile(self, ast_root: mp_ast.Program, symbol_table: 'SymbolTable') -> str:
        self.logger.info(f"Starting C code generation for Program node.")
        self.c_code_lines = [] # Reset for new compilation
        self.indent_level = 0
        self.symbol_table = symbol_table # Store for potential lookups
        
        function_definitions = [defn for defn in ast_root.definitions if isinstance(defn, mp_ast.FuncDef)]
        main_code_definitions = [defn for defn in ast_root.definitions if not isinstance(defn, mp_ast.FuncDef)]

        self._add_line("#include <stdio.h>")
        self._add_line("#include <stdint.h>") # For specific integer types
        self._add_line("#include <stdbool.h>") # For bool type
        self._add_line("#include <stdlib.h>") # For exit()
        self._add_line("") # Blank line
        
        self._generate_helper_functions()
        self._add_line("")

        for func_def in function_definitions:
            self._visit(func_def)
            self._add_line("") 

        self._add_line("int main() {")
        self.indent_level += 1
        
        for definition in main_code_definitions:
            if isinstance(definition, (mp_ast.FuncCall)): # Top-level function call
                result = self._visit(definition)
                self._add_line(result + ";")
            elif isinstance(definition, (mp_ast.VarDef, mp_ast.Assignment, mp_ast.ReturnStmt,
                                        mp_ast.IfStmt, mp_ast.WhileStmt, mp_ast.ForStmt,
                                        mp_ast.IncrementStmt, mp_ast.DecrementStmt,
                                        mp_ast.SelfSetStmt)): # Replaced VarDecl with VarDef
                self._visit(definition) # Statement visitors add their own lines
            elif isinstance(definition, mp_ast.VariableRef): # Check for disallowed VariableRef
                symbol = self.symbol_table.lookup(definition.name, definition) 
                if symbol and symbol.kind == 'function':
                    # Allow void function calls handled by semantic analysis check
                    # Raise error only if semantic analysis didn't (shouldn't happen?)
                    if symbol.attributes.get('return_type') != 'void':
                        raise CompilerError(f"Cannot use non-void function '{definition.name}' as a standalone statement (at {definition.line}:{definition.col})")
                else:
                    # If it's a variable reference, it's still an error as a standalone statement
                    raise CompilerError(f"Cannot use variable '{definition.name}' as a standalone statement (at {definition.line}:{definition.col})")
            else:
                # Raise error for any other unexpected node type at top level of main
                 raise CompilerError(f"Unexpected node type '{type(definition).__name__}' found at top level of main execution (at {definition.line}:{definition.col})")
                
        self._add_line("return 0;") # Default return for main
        self.indent_level -= 1
        self._add_line("}")

        final_c_code = "\n".join(self.c_code_lines)
        self.logger.info(f"Finished C code generation. Total lines: {len(self.c_code_lines)}")
        return final_c_code

    # Visitor Methods

    def visit_IntegerLiteral(self, node: mp_ast.IntegerLiteral) -> str:
        self.logger.debug(f"Processing IntegerLiteral({node.value})")
        return str(node.value)

    def visit_FloatLiteral(self, node: mp_ast.FloatLiteral) -> str:
        self.logger.debug(f"Processing FloatLiteral({node.value})")
        val_str = str(node.value)
        if '.' not in val_str and 'e' not in val_str.lower():
            val_str += ".0" # Make sure it's treated as floating point
        return val_str

    def visit_Assignment(self, node: mp_ast.Assignment) -> str:
        self.logger.debug(f"Processing Assignment(var={node.var_name})")
        var_name = node.var_name
        expression_c = self._visit(node.expression)
        self._add_line(f"{var_name} = {expression_c};")
        # Statement visitor adds line directly
        
    def visit_VariableRef(self, node: mp_ast.VariableRef) -> str:
        self.logger.debug(f"Processing VariableRef({node.name})")
        return node.name

    def visit_BooleanLiteral(self, node: mp_ast.BooleanLiteral) -> str:
        self.logger.debug(f"Processing BooleanLiteral({node.value})")
        return "true" if node.value else "false"
        
    def visit_UnaryOp(self, node: mp_ast.UnaryOp) -> str:
        self.logger.debug(f"Processing UnaryOp(op={node.operator})")
        operand_c = self._visit(node.operand)
        op_map = {
            'not': '!'
            # Add other unary ops if needed
        }
        c_op = op_map.get(node.operator)
        if not c_op:
            raise CompilerError(f"Unsupported unary operator '{node.operator}' at line {node.line}")
        # Add parentheses for safety, especially if operand is complex
        return f"({c_op}{operand_c})"
        
    def visit_BinaryOp(self, node: mp_ast.BinaryOp) -> str:
        self.logger.debug(f"Processing BinaryOp(op={node.operator})")
        left_c = self._visit(node.left)
        right_c = self._visit(node.right)

        # Map language ops to C ops
        op_map = {
            # Arithmetic (assuming correct types for now)
            'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
            'fadd': '+', 'fsub': '-', 'fmul': '*', 'fdiv': '/', 
            # Comparison
            'eq': '==', 'neq': '!=',
            'lt': '<', 'lte': '<=',
            'gt': '>', 'gte': '>=',
            # Logical
            'and': '&&', 'or': '||',
            # Bitwise (assuming XOR is bitwise for integers)
            'xor': '^'
            # Add modulo or other ops if needed
        }
        
        c_op = op_map.get(node.operator)
        if not c_op:
            raise CompilerError(f"Unsupported binary operator '{node.operator}' at line {node.line}")
        
        # Wrap in parentheses for operator precedence safety
        return f"({left_c} {c_op} {right_c})"
        
    def visit_IfStmt(self, node: mp_ast.IfStmt):
        self.logger.debug("Processing IfStmt")
        condition_c = self._visit(node.condition)
        
        # If block
        self._add_line(f"if ({condition_c}) {{")
        self.indent_level += 1
        for stmt in node.if_body:
            result = self._visit(stmt)
            if isinstance(result, str): # Handle expression statements
                 self._add_line(result + ";")
        self.indent_level -= 1
        self._add_line("}")

        # Elif blocks (optional)
        if node.elif_clauses:
            for elif_condition, elif_body in node.elif_clauses:
                elif_condition_c = self._visit(elif_condition)
                self._add_line(f"else if ({elif_condition_c}) {{")
                self.indent_level += 1
                for stmt in elif_body:
                    result = self._visit(stmt)
                    if isinstance(result, str): # Handle expression statements
                         self._add_line(result + ";")
                self.indent_level -= 1
                self._add_line("}")

        # Else block (optional)
        if node.else_body:
            self._add_line("else {")
            self.indent_level += 1
            for stmt in node.else_body:
                result = self._visit(stmt)
                if isinstance(result, str): # Handle expression statements
                     self._add_line(result + ";")
            self.indent_level -= 1
            self._add_line("}")
        # This visitor adds lines directly, doesn't return a string

    def visit_WhileStmt(self, node: mp_ast.WhileStmt):
        self.logger.debug("Processing WhileStmt")
        condition_c = self._visit(node.condition)

        self._add_line(f"while ({condition_c}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self._visit(stmt) # Assume statement visitors add their own lines
        self.indent_level -= 1
        self._add_line("}")
        # This visitor adds lines directly, doesn't return a string

    def visit_SelfSetStmt(self, node: mp_ast.SelfSetStmt):
        self.logger.debug(f"Processing SelfSetStmt(var={node.var_name})")
        var_name = node.var_name
        expression_c = self._visit(node.expression)
        self._add_line(f"{var_name} += {expression_c};") # Use C's += operator
        # This visitor adds lines directly

    def visit_IncrementStmt(self, node: mp_ast.IncrementStmt):
        self.logger.debug(f"Processing IncrementStmt(var={node.var_name})")
        if node.delta is None:
            self._add_line(f"{node.var_name}++;") # Use C's ++ operator for default case
        else:
            delta_c = self._visit(node.delta)
            self._add_line(f"{node.var_name} += {delta_c};") # Use += for increment by value
        # This visitor adds lines directly
        
    def visit_DecrementStmt(self, node: mp_ast.DecrementStmt):
        self.logger.debug(f"Processing DecrementStmt(var={node.var_name})")
        if node.delta is None:
            self._add_line(f"{node.var_name}--;") # Use C's -- operator for default case
        else:
            delta_c = self._visit(node.delta)
            self._add_line(f"{node.var_name} -= {delta_c};") # Use -= for decrement by value
        # This visitor adds lines directly

    def visit_ReturnStmt(self, node: mp_ast.ReturnStmt):
        self.logger.debug(f"Processing ReturnStmt")
        if node.expression:
            expression_c = self._visit(node.expression)
            self._add_line(f"return {expression_c};")
        else:
            self._add_line("return;") # Return void/unit
        # This visitor adds lines directly

    def visit_ForStmt(self, node: mp_ast.ForStmt):
        self.logger.debug(f"Processing ForStmt(var={node.iterator_var})")
        iterator_var = node.iterator_var
        # Assume the iterable is an integer expression representing the upper bound (exclusive)
        iterable_c = self._visit(node.iterable)

        # Declare iterator inside the loop header (requires C99+)
        self._add_line(f"for (int {iterator_var} = 0; {iterator_var} < {iterable_c}; ++{iterator_var}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self._visit(stmt)
        self.indent_level -= 1
        self._add_line("}")
        # This visitor adds lines directly

    def visit_VarDef(self, node: mp_ast.VarDef):
        keyword = "init" if node.is_mutable else "let"
        self.logger.debug(f"Processing VarDef ({keyword}) (type={node.var_type}, var={node.var_name})")
        c_type = C_TYPE_MAP.get(node.var_type)
        if not c_type:
            raise CompilerError(f"Unknown type '{node.var_type}' used in {keyword} statement at line {node.line}")
            
        var_name = node.var_name
        expression_c = self._visit(node.initial_expression)
        
        self._add_line(f"{c_type} {var_name} = {expression_c};")
        # This visitor adds lines directly

    def visit_FuncDef(self, node: mp_ast.FuncDef):
        self.logger.debug(f"Processing FuncDef(name={node.name})")
        # Get C return type
        return_c_type = C_TYPE_MAP.get(node.return_type)
        if not return_c_type:
            raise CompilerError(f"Unknown return type '{node.return_type}' for function '{node.name}'")
        
        # Format parameters
        params_c = []
        for param in node.params:
            param_c_type = C_TYPE_MAP.get(param.type_name)
            if not param_c_type:
                raise CompilerError(f"Unknown parameter type '{param.type_name}' for parameter '{param.name}' in function '{node.name}'")
            params_c.append(f"{param_c_type} {param.name}")
        
        params_str = ", ".join(params_c) if params_c else "void" # Use void for no parameters
        
        self._add_line(f"{return_c_type} {node.name}({params_str}) {{")
        self.indent_level += 1
        
        # Visit statements in the function body
        for stmt in node.body:
            self._visit(stmt)
            
        # Add implicit return 0 if function is main and has no return
        # Or check if void functions need an explicit return? (C allows falling off)
        # For now, assume explicit return is needed if not void

        self.indent_level -= 1
        self._add_line("}")
        # This visitor adds lines directly

    def visit_FuncCall(self, node: mp_ast.FuncCall) -> str:
        self.logger.debug(f"Processing FuncCall (expr={node.func_expr})")

        # Check if the function being called is one of our built-in prints
        if isinstance(node.func_expr, mp_ast.VariableRef):
            func_name = node.func_expr.name
            # Use standard C format specifiers mapped to the types
            # Use double backslash for newline to ensure it's escaped in the final C string
            print_formats = {
                "print_int8":    "%d\\n",   # Assuming fits in int
                "print_int16":   "%d\\n",   # Assuming fits in int
                "print_int32":   "%d\\n",   # Assuming maps to int
                "print_int64":   "%lld\\n", # Assuming maps to long long
                "print_uint8":   "%u\\n",   # Assuming fits in unsigned int
                "print_uint16":  "%u\\n",   # Assuming fits in unsigned int
                "print_uint32":  "%u\\n",   # Assuming maps to unsigned int
                "print_uint64":  "%llu\\n", # Assuming maps to unsigned long long
                "print_float32": "%f\\n",
                "print_float64": "%lf\\n",
                "print_bool":    "%s\\n",
                "print_rune":    "%c\\n",
                "print_string":  "%s\\n"
            }
            
            if func_name in print_formats:
                if len(node.arguments) != 1:
                     raise CompilerError(f"Built-in function '{func_name}' expects 1 argument, got {len(node.arguments)}")
                
                arg_c = self._visit(node.arguments[0])
                # Get the format string *without* extra quotes
                format_str_val = print_formats[func_name]
                
                # Special handling for bool: convert to "true"/"false" string
                if func_name == "print_bool":
                    # Use ternary operator in C: (condition ? "true" : "false")
                    # Escaped quotes needed for the C strings "true" and "false"
                    bool_arg_str = f'({arg_c} ? \\"true\\" : \\"false\\")'
                    # Construct the C printf call string: printf("%s\\n", (arg ? "true" : "false"))
                    return f'printf("{format_str_val}", {bool_arg_str})'
                else:
                    # Construct the C printf call string: printf("format", arg)
                    return f'printf("{format_str_val}", {arg_c})'
            else:
                 # --- Normal Function Call ---
                 # Compile the function expression (usually just a name)
                 func_c = self._visit(node.func_expr) 
                 # Compile arguments
                 args_c = [self._visit(arg) for arg in node.arguments]
                 args_str = ", ".join(args_c)
                 return f"{func_c}({args_str})"
        else:
            # Handle calls where the function itself is an expression (less common)
            # Example: (get_adder()) 5 3 -> Not directly supported in this syntax
            raise CompilerError(f"Calling complex expressions as functions is not yet supported (line {node.line})")

    def visit_ReadIntExpr(self, node: mp_ast.ReadIntExpr) -> str:
        self.logger.debug("Processing ReadIntExpr")
        # Calls a predefined helper function in C
        # Assumes int32_t for now based on the name
        return "mp_read_int32()" # Name of the C helper function

    def _generate_helper_functions(self):
        self.logger.debug("Generating C helper functions")
        # Add helper C functions needed by the compiled code
        
        # --- read_int32 Helper ---
        # Assumes return type matches int32 mapping
        read_type = C_TYPE_MAP.get('int32', 'int') # Default to int if not found
        self._add_line(f"{read_type} mp_read_int32() {{")
        self.indent_level += 1
        self._add_line(f"{read_type} value;")
        # Use fprintf to print prompt to stderr
        self._add_line('fprintf(stderr, "> "); // Prompt for input on stderr') # Use single quotes for Python string
        # Check scanf return value for error handling
        # Use %d for standard int, assuming int32_t maps to int. Adjust if needed.
        self._add_line('if (scanf("%d", &value) != 1) { // Check scanf result')
        self.indent_level += 1
        # Use single quotes for Python string and ensure \n is escaped for C
        self._add_line('fprintf(stderr, "Error reading int32 input.\\n");')
        self._add_line('exit(1); // Exit if input fails')
        self.indent_level -= 1
        self._add_line('}')
        self._add_line("return value;")
        self.indent_level -= 1
        self._add_line("}")
        # Add other helpers here (e.g., read_float64, read_string)

    def visit_BreakStmt(self, node: mp_ast.BreakStmt):
        self.logger.debug("Processing BreakStmt")
        # Ensure break is only used inside loop? (Parser should enforce context if needed)
        self._add_line("break;")
        # This visitor adds lines directly

    def visit_ContinueStmt(self, node: mp_ast.ContinueStmt):
        self.logger.debug("Processing ContinueStmt")
        # Ensure continue is only used inside loop? (Parser should enforce context if needed)
        self._add_line("continue;")
        # This visitor adds lines directly


def main():
    import argparse # Use argparse for better CLI handling

    parser_cli = argparse.ArgumentParser(description='Compile Monopressed code to C.')
    parser_cli.add_argument('input_file', help='Path to the Monopressed (.mp) input file.')
    parser_cli.add_argument("-o", "--output", help="Path to the output C file (default: based on input file)")
    parser_cli.add_argument("--lexer-log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for the lexer (default: INFO)")
    parser_cli.add_argument("--parser-log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for the parser (default: INFO)")
    parser_cli.add_argument("--compiler-log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for the compiler (default: INFO)")
    parser_cli.add_argument("--semantic-log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for the semantic analyzer (default: INFO)")

    args = parser_cli.parse_args()

    # --- Basic File Handling ---
    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    output_file = args.output
    if not output_file:
        base = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base}.c"

    # --- Logging Setup based on args ---
    # Configure individual loggers - assumes root logger might be configured elsewhere too
    logging.getLogger('src.lexer').setLevel(getattr(logging, args.lexer_log_level))
    logging.getLogger('src.parser').setLevel(getattr(logging, args.parser_log_level))
    logging.getLogger('compiler').setLevel(getattr(logging, args.compiler_log_level))
    logging.getLogger('semantic').setLevel(getattr(logging, args.semantic_log_level))

    # --- Compilation Pipeline ---
    try:
        # 1. Read Source Code
        with open(input_file, 'r') as f:
            source_code = f.read()

        # 2. Lexing
        log.info(f"Compiling {input_file} to {output_file}")
        log.info("Starting Lexing...")
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        log.info("Lexing complete.")
        log.debug(f"Tokens: {tokens}")
        
        # 3. Parsing
        log.info("Starting Parsing...")
        parser = Parser(tokens)
        ast = parser.parse()
        log.info("Parsing complete.")
        log.debug(f"AST: {ast}")

        # 4. Semantic Analysis 
        log.info("Starting Semantic Analysis...")
        analyzer = SemanticAnalyzer()
        symbol_table = analyzer.analyze(ast)
        log.info("Semantic Analysis complete.")

        # 5. Code Generation (Compilation)
        log.info("Starting Compilation (AST -> C code)...")
        compiler = Compiler()
        c_code = compiler.compile(ast, symbol_table)
        log.info("Compilation complete.")
        log.debug(f"Generated C Code:\\n------\\n{c_code}\\n------")

        # 6. Write Output File
        with open(output_file, 'w') as f:
            f.write(c_code)
        log.info(f"Successfully wrote C code to {output_file}")

    except (FileNotFoundError, IndentationError, ParseError, SemanticError, CompilerError) as e:
        # Log specific errors caught from pipeline stages
        log.critical(f"{type(e).__name__}: {e}")
        print(f"Error: {e}", file=sys.stderr)
        print("Error: Python compiler (src.compiler) failed.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        log.critical(f"An unexpected error occurred: {e}", exc_info=True) # Log stack trace
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        print("Error: Python compiler (src.compiler) failed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 