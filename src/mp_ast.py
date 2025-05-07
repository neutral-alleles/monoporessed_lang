# src/ast.py
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Tuple

# Basic AST Node Protocol (optional, for type hinting)
class ASTNode:
    # Base class or protocol for all AST nodes
    # Add optional position tracking to the base class
    line: Optional[int] = field(default=None, kw_only=True)
    col: Optional[int] = field(default=None, kw_only=True)
    end_line: Optional[int] = field(default=None, kw_only=True)
    end_col: Optional[int] = field(default=None, kw_only=True)

    def set_pos(self, line: Optional[int], col: Optional[int], end_line: Optional[int] = None, end_col: Optional[int] = None):
        """Helper to set position, useful for nodes spanning multiple tokens."""
        self.line = line
        self.col = col
        self.end_line = end_line if end_line is not None else line
        self.end_col = end_col if end_col is not None else col

# --- Literals ---
@dataclass
class IntegerLiteral(ASTNode):
    value: int
    line: int
    col: int
    def __repr__(self):
        return f"Int({self.value} @{self.line}:{self.col})"

@dataclass
class FloatLiteral(ASTNode):
    value: float
    line: int
    col: int
    def __repr__(self):
        return f"Float({self.value} @{self.line}:{self.col})"

@dataclass
class BooleanLiteral(ASTNode):
    value: bool
    line: int
    col: int
    def __repr__(self):
        return f"Bool({self.value} @{self.line}:{self.col})"

@dataclass
class RuneLiteral(ASTNode):
    value: str # Store as the single character string
    line: int
    col: int
    def __repr__(self):
        return f"Rune({repr(self.value)} @{self.line}:{self.col})"

@dataclass
class StringLiteral(ASTNode):
    value: str
    line: int
    col: int
    def __repr__(self):
        # Use explicit double quotes around the raw value
        return f'String("{self.value}" @{self.line}:{self.col})'

# --- Expressions ---
@dataclass
class VariableRef(ASTNode):
    name: str
    line: int
    col: int
    def __repr__(self):
        return f"VarRef({self.name} @{self.line}:{self.col})"

@dataclass
class BinaryOp(ASTNode):
    operator: str # The keyword acting as operator (e.g., "add")
    left: 'Expression' # Use forward reference
    right: 'Expression' # Use forward reference
    line: int # Line of the operator keyword
    col: int # Col of the operator keyword
    def __repr__(self):
        return f"Op({self.operator} {self.left} {self.right} @{self.line}:{self.col})"

@dataclass
class UnaryOp(ASTNode):
    operator: str # e.g., "not"
    operand: 'Expression' # Use forward reference
    line: int # Line of the operator keyword
    col: int # Col of the operator keyword
    def __repr__(self):
        return f"Op({self.operator} {self.operand} @{self.line}:{self.col})"

# Update Expression to include BinaryOp and UnaryOp
Expression = IntegerLiteral | FloatLiteral | BooleanLiteral | RuneLiteral | StringLiteral | VariableRef | BinaryOp | UnaryOp

# --- Statements ---
@dataclass
class Assignment(ASTNode):
    var_name: str # Identifier being assigned to
    expression: Expression # The value/expression being assigned
    line: int # Line of the 'set' keyword
    col: int # Col of the 'set' keyword
    def __repr__(self):
        return f"Assign({self.var_name} = {self.expression} @{self.line}:{self.col})"

@dataclass
class ReturnStmt(ASTNode):
    expression: Optional[Expression]  # Can be None for 'return' with no value
    line: int # Line of the 'return' keyword
    col: int # Col of the 'return' keyword
    def __repr__(self):
        expr_repr = repr(self.expression) if self.expression is not None else "None"
        # Ensure line/col are included, matching other nodes
        return f"Return({expr_repr} @{self.line}:{self.col})"

@dataclass
class IfStmt(ASTNode):
    condition: 'Expression'
    if_body: List['Statement']
    # Add elif clauses: list of (condition, body) pairs
    elif_clauses: Optional[List[Tuple['Expression', List['Statement']]]] = field(default=None) 
    else_body: Optional[List['Statement']] = field(default=None) # Optional else block
    line: int # Line of 'if' keyword
    col: int # Col of 'if' keyword
    def __repr__(self):
        # Always include else_body, show None explicitly if it's not present
        elif_repr = "" # Start empty
        if self.elif_clauses:
            elif_repr_parts = []
            for cond, body in self.elif_clauses:
                 elif_repr_parts.append(f"elif({repr(cond)}, then={repr(body)})")
            elif_repr = ", " + ", ".join(elif_repr_parts)
        
        else_repr = f", else={repr(self.else_body)}"
        return f"If({repr(self.condition)}, then={repr(self.if_body)}{elif_repr}{else_repr} @{self.line}:{self.col})"

@dataclass
class WhileStmt(ASTNode):
    condition: 'Expression'
    body: List['Statement']
    line: int # Line of 'while' keyword
    col: int # Col of 'while' keyword
    def __repr__(self):
        return f"While({self.condition}, body={self.body} @{self.line}:{self.col})"

@dataclass
class ForStmt(ASTNode):
    # Simple form: for <var> in range <N>
    iterator_var: str
    iterable: 'Expression' # Expected to be an integer expression for now
    body: List['Statement']
    line: int # Line of 'for' keyword
    col: int # Col of 'for' keyword
    def __repr__(self):
        return f"For({self.iterator_var} in {repr(self.iterable)}, body={repr(self.body)} @{self.line}:{self.col})"

@dataclass
class IncrementStmt(ASTNode):
    var_name: str
    delta: Optional['Expression'] # The amount to increment by (defaults to 1 if None)
    line: int # Line of 'inc' keyword
    col: int # Col of 'inc' keyword
    def __repr__(self):
        if self.delta is None:
            # Represent the default case (increment by 1)
            return f"Inc({self.var_name} @{self.line}:{self.col})"
        else:
            # Represent increment by a specific expression
            return f"Inc({self.var_name} by {repr(self.delta)} @{self.line}:{self.col})"

@dataclass
class DecrementStmt(ASTNode):
    var_name: str
    delta: Optional['Expression'] # The amount to decrement by (defaults to 1 if None)
    line: int # Line of 'dec' keyword
    col: int # Col of 'dec' keyword
    def __repr__(self):
        if self.delta is None:
            # Represent the default case (decrement by 1)
            return f"Dec({self.var_name} @{self.line}:{self.col})"
        else:
            # Represent decrement by a specific expression
            return f"Dec({self.var_name} by {repr(self.delta)} @{self.line}:{self.col})"

@dataclass
class SelfSetStmt(ASTNode):
    """Represents 'selfset x y' which is syntactic sugar for 'set x add x y'"""
    var_name: str
    # Operator field removed - implicitly 'add'
    expression: 'Expression'
    line: int
    col: int
    def __repr__(self):
        # Updated repr to reflect implicit add
        return f"SelfSet({self.var_name} += {repr(self.expression)} @{self.line}:{self.col})"

@dataclass
class VarDef(ASTNode):
    """Represents 'init|let <type> <name> <expression>'."""
    is_mutable: bool # True if declared with 'init', False if 'let'
    var_type: str
    var_name: str
    initial_expression: Expression
    line: int # Line of 'init' or 'let' keyword
    col: int # Col of 'init' or 'let' keyword
    def __repr__(self):
        keyword = "init" if self.is_mutable else "let"
        return f"{keyword}({self.var_type} {self.var_name} = {repr(self.initial_expression)} @{self.line}:{self.col})"

# --- Loop Control Statements ---
@dataclass
class BreakStmt(ASTNode):
    """Represents a 'break' statement."""
    line: int # Line of 'break' keyword
    col: int # Col of 'break' keyword
    def __repr__(self):
        return f"Break(@{self.line}:{self.col})"

@dataclass
class ContinueStmt(ASTNode):
    """Represents a 'continue' statement."""
    line: int # Line of 'continue' keyword
    col: int # Col of 'continue' keyword
    def __repr__(self):
        return f"Continue(@{self.line}:{self.col})"

# Update Statement to include control flow and inc/dec/selfset
Statement = Assignment | ReturnStmt | IfStmt | WhileStmt | ForStmt | IncrementStmt | DecrementStmt | SelfSetStmt | VarDef | BreakStmt | ContinueStmt

# --- Program Structure / Definitions ---

@dataclass
class FuncParam(ASTNode):
    """Represents a function parameter."""
    name: str
    type_name: str # Store the type name as string
    line: int
    col: int
    def __repr__(self):
        return f"Param({self.name}: {self.type_name} @{self.line}:{self.col})"

@dataclass
class FuncDef(ASTNode):
    """Represents a function definition."""
    name: str
    params: List[FuncParam]
    return_type: str # Store return type name as string
    body: List[Statement] # Body is a list of statements
    line: int # Line of 'def' keyword
    col: int # Col of 'def' keyword
    def __repr__(self):
        params_repr = ", ".join(repr(p) for p in self.params)
        return f"FuncDef({self.name}({params_repr}) -> {self.return_type} body=[...] @{self.line}:{self.col})"

@dataclass
class FuncCall(ASTNode):
    """Represents a function call like 'func arg1 arg2 ...'."""
    func_expr: Expression # Function name/expression (e.g., VariableRef('fib'))
    arguments: List[Expression]
    line: int # Line of the function identifier/expression
    col: int # Col of the function identifier/expression
    def __repr__(self):
        args_repr = ", ".join(repr(arg) for arg in self.arguments)
        # Simple repr for now, assuming func_expr is usually a VarRef
        func_name = repr(self.func_expr) 
        # Basic check to simplify common case VarRef('name') -> name
        if isinstance(self.func_expr, VariableRef):
            func_name = self.func_expr.name
        return f"FuncCall({func_name}({args_repr}) @{self.line}:{self.col})"

@dataclass
class ReadIntExpr(ASTNode):
    """Represents 'read_int32()' expression (assuming int32 for now)"""
    line: int # Line of 'read_int32' keyword
    col: int # Col of 'read_int32' keyword
    def __repr__(self):
        return f"ReadIntExpr(@{self.line}:{self.col})"

@dataclass
class Program(ASTNode):
    """Root node for the entire program."""
    # Allow top-level statements OR function definitions OR standalone func calls
    definitions: List[Union[Statement, FuncDef, FuncCall]] 
    # Inherits position attributes from ASTNode
    def __repr__(self):
        defs_repr = ", ".join(repr(d) for d in self.definitions)
        return f"Program([{defs_repr}])"

# Update Expression union to include FuncCall and ReadIntExpr
Expression = IntegerLiteral | FloatLiteral | BooleanLiteral | RuneLiteral | StringLiteral | VariableRef | BinaryOp | UnaryOp | FuncCall | ReadIntExpr

# Update Statement union again (Removed PrintStmt)
Statement = Assignment | ReturnStmt | IfStmt | WhileStmt | ForStmt | IncrementStmt | DecrementStmt | SelfSetStmt | VarDef | BreakStmt | ContinueStmt

# Need Tuple for IfStmt elif_clauses type hint
from typing import Tuple

# Update Expression union to include FuncCall and ReadIntExpr
Expression = IntegerLiteral | FloatLiteral | BooleanLiteral | RuneLiteral | StringLiteral | VariableRef | BinaryOp | UnaryOp | FuncCall | ReadIntExpr

# Update Statement union again (Removed PrintStmt)
Statement = Assignment | ReturnStmt | IfStmt | WhileStmt | ForStmt | IncrementStmt | DecrementStmt | SelfSetStmt | VarDef | BreakStmt | ContinueStmt 