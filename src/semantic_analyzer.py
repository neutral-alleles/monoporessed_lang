# src/semantic_analyzer.py
import logging
from typing import Optional, List, Dict, Any, Tuple
from . import mp_ast
from .mp_ast import ASTNode

log = logging.getLogger('semantic')

class SemanticError(Exception):
    """Custom exception for semantic errors found during analysis."""
    def __init__(self, message: str, node: Optional[ASTNode] = None):
        # Add line/col info to message if available
        loc = ""
        if node and hasattr(node, 'line') and node.line is not None and hasattr(node, 'col') and node.col is not None:
            loc = f" (at {node.line}:{node.col})"
        super().__init__(message + loc)
        self.node = node

class Symbol:
    """Represents an entry in the symbol table (variable or function)."""
    def __init__(self, name: str, kind: str, node: ASTNode, **kwargs):
        self.name = name
        self.kind = kind # 'variable', 'function'
        self.node = node # AST node where defined
        # Add other attributes based on kind
        self.attributes = kwargs 
        # --- Add usage tracking ---
        self.is_used: bool = False 
        # Example for variable: type='int32'
        # Example for function: params=['int32', 'bool'], return_type='int32'

    def __repr__(self):
        # Include is_used status in repr
        return f"Symbol({self.name}, kind={self.kind}, used={self.is_used}, attrs={self.attributes})"

class SymbolTable:
    """Manages scopes and symbols (variables, functions) during semantic analysis."""
    def __init__(self):
        # Stack of scopes, each scope is a dictionary: {name: Symbol}
        # Global scope is always at the bottom (index 0)
        self.scope_stack: List[Dict[str, Symbol]] = [{}] 
        log.debug("SymbolTable initialized with global scope.")

    def enter_scope(self):
        """Enter a new nested scope."""
        log.debug(f"Entering new scope (level {len(self.scope_stack)})...")
        self.scope_stack.append({})

    def exit_scope(self):
        """Exit the current scope."""
        if len(self.scope_stack) <= 1:
            log.error("Attempted to exit global scope.") # Should not happen
            return 
        log.debug(f"Exiting scope (level {len(self.scope_stack) - 1})...")
        self.scope_stack.pop()

    def _current_scope(self) -> Dict[str, Symbol]:
        return self.scope_stack[-1]

    def define(self, symbol: Symbol):
        """Define a new symbol in the current scope."""
        current_scope = self._current_scope()
        name = symbol.name
        log.debug(f"Defining symbol in current scope: {name} -> {symbol}")
        if name in current_scope:
            existing_symbol = current_scope[name]
            raise SemanticError(f"Symbol '{name}' already defined in this scope (at {existing_symbol.node.line}:{existing_symbol.node.col})", symbol.node)
        current_scope[name] = symbol

    def lookup(self, name: str, node: Optional[ASTNode] = None) -> Optional[Symbol]:
        """Look up a symbol's type starting from the current scope outwards."""
        log.debug(f"Looking up symbol: {name}")
        # Search from innermost scope outwards
        for scope in reversed(self.scope_stack):
            if name in scope:
                symbol = scope[name]
                log.debug(f"  Found '{name}' in scope level {self.scope_stack.index(scope)}: {symbol}")
                return symbol
        
        # If not found in any scope
        raise SemanticError(f"Symbol '{name}' not defined", node)
        
class SemanticAnalyzer:
    """Performs semantic analysis, including type checking, on the AST."""
    def __init__(self):
        self.symbol_table = SymbolTable()
        # --- Add loop depth tracking ---
        self.loop_depth = 0 
        # --- Add state for current function analysis ---
        self.current_function_return_type: Optional[str] = None
        log.debug("SemanticAnalyzer initialized.")

    def analyze(self, node: ASTNode) -> SymbolTable:
        """Entry point for semantic analysis. Returns the populated SymbolTable on success."""
        log.info(f"Starting semantic analysis from node: {type(node).__name__}")
        try:
            self._visit(node)
            log.info("Semantic analysis completed successfully.")
            return self.symbol_table # <<< Return symbol table on success
        except SemanticError as e:
            log.error(f"Semantic Error: {e}")
            # Re-raise to signal failure to the caller (e.g., the compiler main)
            raise e 
        except Exception as e:
            log.error(f"Unexpected error during semantic analysis: {e}", exc_info=True)
            raise # Re-raise unexpected errors

    def _visit(self, node: ASTNode) -> Optional[str]:
        """Dispatch to the appropriate visit method. Returns the node's type if applicable."""
        if node is None: 
            log.warning("Attempted to visit a None node.")
            return None # Or raise error?
            
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        log.debug(f"Visiting {type(node).__name__} node using {visitor.__name__}")
        
        # Visitor methods should return the *type* of the expression node they visit,
        # or None for statements.
        node_type = visitor(node)
        
        if node_type is not None:
            log.debug(f"  -> Node type determined: {node_type}")
            
        return node_type

    def generic_visit(self, node: ASTNode):
        """Fallback for unhandled AST node types."""
        log.warning(f"No specific visit method defined for node type: {type(node).__name__}")
        # For nodes with children, we might want to visit them by default?
        # For now, raise error or just log and return None
        # raise NotImplementedError(f"No visit_{type(node).__name__} method")
        return None # Default: No type determined / Statement node

    # --- Visitor Methods (Initial Placeholders) ---

    def visit_Program(self, node: mp_ast.Program):
        log.debug("Analyzing Program node")
        # Analyze all definitions
        for definition in node.definitions:
            # Handle different top-level node types
            if isinstance(definition, mp_ast.FuncDef):
                self._visit(definition)
            elif isinstance(definition, mp_ast.VarDef): # Check for VarDef (covers let/init)
                self._visit(definition)
            elif isinstance(definition, mp_ast.Assignment): 
                self._visit(definition)
            elif isinstance(definition, mp_ast.VariableRef):
                # Check if a variable/function name is used standalone
                symbol = self.symbol_table.lookup(definition.name, definition)
                if symbol.kind == 'function':
                    return_type = symbol.attributes.get('return_type')
                    if return_type != 'void':
                        raise SemanticError(f"Cannot use function '{definition.name}' with non-void return type '{return_type}' as a standalone statement. Did you mean to assign its result?")
                    else:
                        log.debug(f"Standalone call to void function '{definition.name}' is allowed.")
                        symbol.is_used = True
                else: # It's a variable
                    raise SemanticError(f"Cannot use variable '{definition.name}' as a standalone statement")
            elif isinstance(definition, mp_ast.FuncCall):
                # Analyze the function call (checks types, arity, marks used, etc.)
                self._visit(definition)
            else: 
                # Visit other valid statement types (e.g., IfStmt, WhileStmt at top level?)
                self._visit(definition)
                
        self._check_unused_symbols("global scope")

        return None # Program node itself doesn't have a type

    # -- Literals --
    def visit_IntegerLiteral(self, node: mp_ast.IntegerLiteral) -> str:
        # For now, assume all integer literals fit into int32 for simplicity
        # TODO: Refine based on value range?
        return 'int32'

    def visit_FloatLiteral(self, node: mp_ast.FloatLiteral) -> str:
        # For now, assume all float literals are float64
        # TODO: Refine based on precision or suffix?
        return 'float64'

    def visit_BooleanLiteral(self, node: mp_ast.BooleanLiteral) -> str:
        return 'bool'

    def visit_RuneLiteral(self, node: mp_ast.RuneLiteral) -> str:
        return 'rune'

    def visit_StringLiteral(self, node: mp_ast.StringLiteral) -> str:
        return 'string'

    # -- Variable related --
    def visit_VariableRef(self, node: mp_ast.VariableRef) -> str:
        log.debug(f"Analyzing VariableRef: {node.name}")
        symbol = self.symbol_table.lookup(node.name, node)
        if symbol.kind != 'variable':
            raise SemanticError(f"Identifier '{node.name}' is not a variable (it's a {symbol.kind})", node)
        log.debug(f"Marking variable symbol '{symbol.name}' as used.")
        symbol.is_used = True
        return symbol.attributes.get('type')
        
    def visit_Assignment(self, node: mp_ast.Assignment):
        log.debug(f"Analyzing Assignment: {node.var_name} = ...")
        # 1. Check if variable exists and is mutable
        symbol = self.symbol_table.lookup(node.var_name, node)
        if symbol.kind != 'variable':
             raise SemanticError(f"Cannot assign to '{node.var_name}' because it is not a variable (it's a {symbol.kind})", node)
        
        # --- Check Mutability --- 
        if not symbol.attributes.get('is_mutable', False): # Default to immutable if flag missing?
            raise SemanticError(f"Cannot assign to immutable variable '{node.var_name}' (declared with 'let')", node)
            
        var_type = symbol.attributes.get('type')
        
        # 2. Visit the expression to get its type
        expr_type = self._visit(node.expression)
        if expr_type is None: 
             raise SemanticError(f"Could not determine type of expression in assignment to '{node.var_name}'", node.expression)
             
        # 3. Check for type compatibility
        if var_type != expr_type:
            raise SemanticError(f"Type mismatch: Cannot assign type '{expr_type}' to variable '{node.var_name}' of type '{var_type}'", node)
            
        log.debug(f"Assignment type check passed for mutable var: {node.var_name} ({var_type}) = ... ({expr_type})")
        return None # Statement node

    def visit_VarDef(self, node: mp_ast.VarDef):
        keyword = "init" if node.is_mutable else "let"
        log.debug(f"Analyzing VarDef ({keyword}): {node.var_type} {node.var_name} = ...")
        
        expr_type = self._visit(node.initial_expression)
        if expr_type is None: 
             raise SemanticError(f"Could not determine type of initializer expression for '{node.var_name}'", node.initial_expression)
             
        declared_type = node.var_type
        if declared_type != expr_type:
            raise SemanticError(f"Type mismatch: Cannot initialize variable '{node.var_name}' of type '{declared_type}' with expression of type '{expr_type}'", node)
        
        # Define the variable, storing its mutability
        var_symbol = Symbol(
            node.var_name, 
            kind='variable', 
            type=declared_type, 
            is_mutable=node.is_mutable, # Store mutability flag
            node=node
        )
        self.symbol_table.define(var_symbol)

        log.debug(f"VarDef analysis passed for {node.var_name} (mutable={node.is_mutable})")
        return None # Statement node

    # TODO: Add more visitor methods (BinaryOp, UnaryOp, FuncDef, FuncCall, IfStmt, etc.)

    # --- Operators ---
    def visit_BinaryOp(self, node: mp_ast.BinaryOp) -> str:
        log.debug(f"Analyzing BinaryOp: operator={node.operator}")

        left_type = self._visit(node.left)
        right_type = self._visit(node.right)

        if not left_type or not right_type:
            # This shouldn't happen if operands are valid expressions
            raise SemanticError(f"Could not determine type for one or both operands of '{node.operator}'", node)

        op = node.operator
        
        # Define type sets for checking
        integer_types = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'}
        float_types = {'float32', 'float64'}
        numeric_types = integer_types.union(float_types)
        # comparable_types = numeric_types.union({'bool', 'rune', 'string'}) # Adjust as needed
        
        # Check strict type equality for operands first
        if left_type != right_type:
            raise SemanticError(f"Type mismatch for operator '{op}': operands have different types '{left_type}' and '{right_type}'", node)

        # Type validity checks based on operator category
        result_type = None
        if op in {'add', 'sub', 'mul', 'div', 'xor'}: # Integer Arithmetic/Bitwise
            if left_type not in integer_types:
                raise SemanticError(f"Operator '{op}' requires integer operands, but got '{left_type}'", node)
            result_type = left_type # Result is the same integer type
        elif op in {'fadd', 'fsub', 'fmul', 'fdiv'}: # Float Arithmetic
            if left_type not in float_types:
                raise SemanticError(f"Operator '{op}' requires float operands, but got '{left_type}'", node)
            result_type = left_type # Result is the same float type
        elif op in {'eq', 'neq', 'lt', 'lte', 'gt', 'gte'}: # Comparisons
            # Allow comparison for numeric types, bool, rune, string (if supported)
            # TODO: Refine comparable types if necessary (e.g., add string comparison later)
            if left_type not in numeric_types and left_type not in {'bool', 'rune'}:
                 raise SemanticError(f"Operator '{op}' cannot compare type '{left_type}'", node)
            result_type = 'bool' # Result is always boolean
        elif op in {'and', 'or'}: # Logical
            if left_type != 'bool':
                raise SemanticError(f"Operator '{op}' requires boolean operands, but got '{left_type}'", node)
            result_type = 'bool' # Result is boolean
        else:
            # Should not happen if parser validates operators
            raise SemanticError(f"Unsupported or unknown binary operator '{op}' encountered during semantic analysis", node)

        log.debug(f"BinaryOp '{op}' type check passed. Left: {left_type}, Right: {right_type} -> Result: {result_type}")
        return result_type

    def visit_UnaryOp(self, node: mp_ast.UnaryOp) -> str:
        log.debug(f"Analyzing UnaryOp: operator={node.operator}")
        operand_type = self._visit(node.operand)
        op = node.operator
        
        if not operand_type:
             raise SemanticError(f"Could not determine type for operand of '{op}'", node)

        result_type = None
        if op == 'not':
            if operand_type != 'bool':
                 raise SemanticError(f"Operator '{op}' requires a boolean operand, but got '{operand_type}'", node)
            result_type = 'bool'
        # Add other unary operators (e.g., numeric negation) here if needed
        # elif op == 'neg': ...
        else:
             raise SemanticError(f"Unsupported unary operator '{op}'", node)
             
        log.debug(f"UnaryOp '{op}' type check passed. Operand: {operand_type} -> Result: {result_type}")
        return result_type 

    # --- Control Flow ---
    def visit_IfStmt(self, node: mp_ast.IfStmt):
        log.debug("Analyzing IfStmt")
        # Check condition type
        cond_type = self._visit(node.condition)
        if cond_type != 'bool':
            raise SemanticError(f"If condition must be type 'bool', but got '{cond_type}'", node.condition)
        
        # Analyze 'if' body (TODO: Introduce scope?)
        log.debug("Analyzing 'if' body")
        for stmt in node.if_body:
            self._visit(stmt)
            
        # Analyze 'elif' clauses (TODO: Introduce scope?)
        if node.elif_clauses:
            log.debug("Analyzing 'elif' clauses")
            for elif_cond, elif_body in node.elif_clauses:
                elif_cond_type = self._visit(elif_cond)
                if elif_cond_type != 'bool':
                    raise SemanticError(f"Elif condition must be type 'bool', but got '{elif_cond_type}'", elif_cond)
                log.debug("Analyzing 'elif' body")
                for stmt in elif_body:
                     self._visit(stmt)

        # Analyze 'else' body (TODO: Introduce scope?)
        if node.else_body:
            log.debug("Analyzing 'else' body")
            for stmt in node.else_body:
                self._visit(stmt)
                
        return None # Statement node

    def visit_WhileStmt(self, node: mp_ast.WhileStmt):
        log.debug("Analyzing WhileStmt")
        # Check condition type
        cond_type = self._visit(node.condition)
        if cond_type != 'bool':
            raise SemanticError(f"While condition must be type 'bool', but got '{cond_type}'", node.condition)
            
        # Analyze body (TODO: Introduce scope?)
        log.debug("Analyzing 'while' body")
        # --- Increment loop depth ---
        self.loop_depth += 1
        log.debug(f"Entered loop scope. Depth: {self.loop_depth}")
        for stmt in node.body:
            self._visit(stmt)
        # --- Decrement loop depth ---
        self.loop_depth -= 1
        log.debug(f"Exited loop scope. Depth: {self.loop_depth}")
            
        return None # Statement node

    def visit_ForStmt(self, node: mp_ast.ForStmt):
        log.debug("Analyzing ForStmt")
        # Check iterable type (must be integer for range)
        iterable_type = self._visit(node.iterable)
        integer_types = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'}
        if iterable_type not in integer_types:
             raise SemanticError(f"For loop range expression must be an integer type, but got '{iterable_type}'", node.iterable)
        
        # TODO: Introduce scope for iterator variable
        # Define iterator variable (assuming int32 for now)
        # self.symbol_table.define(node.iterator_var, 'int32', node) 
        
        # Analyze body (TODO: Set loop context for break/continue check)
        log.debug("Analyzing 'for' body")
        # --- Increment loop depth ---
        self.loop_depth += 1
        log.debug(f"Entered loop scope. Depth: {self.loop_depth}")
        for stmt in node.body:
            self._visit(stmt)
        # --- Decrement loop depth ---
        self.loop_depth -= 1
        log.debug(f"Exited loop scope. Depth: {self.loop_depth}")
            
        # TODO: Exit iterator scope
        return None # Statement node
        
    # --- Add visit_BreakStmt/visit_ContinueStmt (placeholder/basic check) ---
    def visit_BreakStmt(self, node: mp_ast.BreakStmt):
        log.debug("Analyzing BreakStmt")
        # Check if inside a loop using context/state
        if self.loop_depth <= 0:
           raise SemanticError("'break' statement not allowed outside of a loop", node)
        log.debug("Break statement is inside a loop.")
        return None
        
    def visit_ContinueStmt(self, node: mp_ast.ContinueStmt):
        log.debug("Analyzing ContinueStmt")
        # Check if inside a loop using context/state
        if self.loop_depth <= 0:
           raise SemanticError("'continue' statement not allowed outside of a loop", node)
        log.debug("Continue statement is inside a loop.")
        return None

    # --- Functions & Scoping ---
    def visit_FuncDef(self, node: mp_ast.FuncDef):
        log.debug(f"Analyzing FuncDef: {node.name}")

        # 1. Define function symbol in the *current* (parent) scope
        param_types = [p.type_name for p in node.params]
        func_symbol = Symbol(node.name, kind='function', node=node,
                             params=param_types, return_type=node.return_type)
        # Check for redefinition in the current scope *before* defining
        # The define method already does this, but being explicit doesn't hurt review
        # if self.symbol_table._current_scope().get(node.name):
        #    raise SemanticError(f"Symbol '{node.name}' already defined in this scope", node)
        self.symbol_table.define(func_symbol) # Define in parent scope

        # --- Store expected return type and enter scope ---
        previous_return_type = self.current_function_return_type # Save outer context
        self.current_function_return_type = node.return_type
        log.debug(f"Set expected return type for '{node.name}' to: {self.current_function_return_type}")

        self.symbol_table.enter_scope() # <<< Enter new scope for function body

        # 3. Define parameters within the function scope
        log.debug(f"Defining parameters for {node.name} in new scope:")
        for param in node.params:
            param_symbol = Symbol(param.name, kind='variable', type=param.type_name, node=param)
            # Define parameter in the function's new scope
            self.symbol_table.define(param_symbol)

        # 4. Visit the function body
        log.debug(f"Analyzing body of function {node.name}...")
        for stmt in node.body:
            self._visit(stmt)

        # --- Add check for unused symbols before exiting scope ---
        self._check_unused_symbols(f"function {node.name}")

        # --- Restore context and exit scope ---
        self.symbol_table.exit_scope() # <<< Exit function scope
        self.current_function_return_type = previous_return_type # Restore outer context
        log.debug(f"Restored outer expected return type: {self.current_function_return_type}")

        log.debug(f"Finished analyzing FuncDef: {node.name}")
        return None # Statement node

    def visit_FuncCall(self, node: mp_ast.FuncCall) -> str:
        log.debug(f"Analyzing FuncCall: {node.func_expr}")
        
        # 1. Ensure function being called is a simple name
        if not isinstance(node.func_expr, mp_ast.VariableRef):
            raise SemanticError("Calling complex expressions as functions not yet supported", node.func_expr)
        
        func_name = node.func_expr.name
        provided_args = node.arguments # Use the correct attribute name
        num_provided_args = len(provided_args)

        # --- 2. Special handling for built-in print/read functions ---
        # Define expected types and return types for built-ins
        builtin_print_types = { # Maps func name to expected arg type
            "print_int8": 'int8', "print_int16": 'int16', "print_int32": 'int32', "print_int64": 'int64',
            "print_uint8": 'uint8', "print_uint16": 'uint16', "print_uint32": 'uint32', "print_uint64": 'uint64',
            "print_float32": 'float32', "print_float64": 'float64',
            "print_bool": 'bool', "print_rune": 'rune', "print_string": 'string',
        }
        builtin_read_types = { # Maps func name to return type
            "read_int32": 'int32', # Add others as needed
            # "read_int64": 'int64', ...
        }

        if func_name in builtin_print_types:
            log.debug(f"Analyzing built-in print function: {func_name}")
            expected_arity = 1
            if num_provided_args != expected_arity:
                raise SemanticError(f"Built-in function '{func_name}' expects {expected_arity} argument(s), but got {num_provided_args}", node)
            
            expected_arg_type = builtin_print_types[func_name]
            actual_arg_type = self._visit(provided_args[0])
            if actual_arg_type != expected_arg_type:
                raise SemanticError(f"Argument type mismatch for '{func_name}': expected '{expected_arg_type}', but got '{actual_arg_type}'", provided_args[0])
            log.debug(f"Built-in '{func_name}' call analysis passed.")
            return 'void' # Print functions don't return a value usable in expressions

        elif func_name in builtin_read_types:
            log.debug(f"Analyzing built-in read function: {func_name}")
            expected_arity = 0
            if num_provided_args != expected_arity:
                raise SemanticError(f"Built-in function '{func_name}' expects {expected_arity} argument(s), but got {num_provided_args}", node)
            
            return_type = builtin_read_types[func_name]
            log.debug(f"Built-in '{func_name}' call analysis passed. Returns: {return_type}")
            return return_type

        # --- 3. Handle User-Defined Functions ---
        log.debug(f"Analyzing user-defined function call: {func_name}")
        symbol = self.symbol_table.lookup(func_name, node.func_expr)
        if symbol.kind != 'function':
            raise SemanticError(f"Identifier '{func_name}' is not a function (it's a {symbol.kind})", node.func_expr)
        
        # --- 4. Check Arity ---
        expected_params = symbol.attributes.get('params', [])
        expected_arity = len(expected_params)
        if num_provided_args != expected_arity:
            raise SemanticError(f"Function '{func_name}' expects {expected_arity} argument(s), but got {num_provided_args}", node)
        
        # --- 5. Check Argument Types ---
        log.debug(f"Checking arguments for {func_name}: Provided={num_provided_args}, Expected={expected_arity}")
        for i, (arg_node, expected_type) in enumerate(zip(provided_args, expected_params)):
            log.debug(f"Checking arg {i+1}: expected type '{expected_type}'")
            actual_type = self._visit(arg_node)
            if actual_type != expected_type:
                raise SemanticError(f"Argument type mismatch for parameter {i+1} of function '{func_name}': expected '{expected_type}', but got '{actual_type}'", arg_node)
        
        # --- 6. Return Function's Declared Return Type ---
        return_type = symbol.attributes.get('return_type')
        if return_type is None:
             # This should not happen if functions are defined correctly
             raise SemanticError(f"Internal error: Could not determine return type for function '{func_name}'", node)
             
        log.debug(f"Function call '{func_name}' type check passed. Returns: {return_type}")
        return return_type

    def visit_ReturnStmt(self, node: mp_ast.ReturnStmt):
        log.debug("Analyzing ReturnStmt")
        
        # 1. Check if we are inside a function
        if self.current_function_return_type is None:
            raise SemanticError("'return' statement found outside of a function", node)
            
        expected_type = self.current_function_return_type
        log.debug(f"Checking return statement. Expected type: {expected_type}")

        if node.expression:
            # 3. Return with a value
            actual_type = self._visit(node.expression)
            if actual_type is None: # Should not happen for valid expressions
                raise SemanticError("Could not determine type of return expression", node.expression)
                
            # Check if expected type is 'void' when a value is returned
            if expected_type == 'void':
                raise SemanticError(f"Function with return type 'void' cannot return a value (got type '{actual_type}')", node)
                
            # Check for type mismatch
            if actual_type != expected_type:
                raise SemanticError(f"Return type mismatch: function expects '{expected_type}', but return expression has type '{actual_type}'", node.expression)
            
            log.debug(f"Return statement type check passed (value return): Expected={expected_type}, Actual={actual_type}")
        else:
            # 2. Return without a value (empty return)
            # Check if expected type is *not* 'void'
            if expected_type != 'void':
                raise SemanticError(f"Empty 'return' used in function expecting return type '{expected_type}'", node)
            
            log.debug(f"Return statement type check passed (empty return for void function).")
            
        return None # Statement node

    # --- Add helper for checking unused symbols ---
    def _check_unused_symbols(self, scope_name: str):
        """Checks for unused variables in the current scope and raises errors."""
        log.debug(f"Checking for unused symbols in {scope_name}...")
        current_scope = self.symbol_table._current_scope()
        for name, symbol in current_scope.items():
            if symbol.kind == 'variable' and not symbol.is_used:
                # Use the definition node for error location
                raise SemanticError(f"Variable '{name}' declared but never used in {scope_name}", symbol.node)
            # Optionally, add checks for unused functions if desired
            # elif symbol.kind == 'function' and not symbol.is_used:
            #    raise SemanticError(f"Function '{name}' defined but never used in {scope_name}", symbol.node)
        log.debug(f"Finished checking unused symbols in {scope_name}.") 