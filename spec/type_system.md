# Monopressed Language - Type System Specification

This document outlines the type system for the Monopressed language.

## Goals

- **Static Typing:** All types are checked before runtime (during compilation).
- **Strictness:** Provide stricter type rules than C to prevent common errors (e.g., disallow implicit conversions between unrelated types where C might allow them).
- **Clarity:** Types are explicitly specified for variables and function signatures.
- **Mutability Control:** Distinguish between mutable and immutable variables.

## Primitive Types

The language supports the following primitive types:

| Monopressed Type | C Equivalent (Typical) | Description                     | Literal Examples           |
|------------------|------------------------|---------------------------------|----------------------------|
| `int8`           | `int8_t`               | 8-bit signed integer            | `-10`, `0`, `127`          |
| `int16`          | `int16_t`              | 16-bit signed integer           | `-30000`, `1000`         |
| `int32`          | `int32_t`              | 32-bit signed integer           | `-2000000000`, `50000`    |
| `int64`          | `int64_t`              | 64-bit signed integer           | `-9e18`, `1e12`           |
| `uint8`          | `uint8_t`              | 8-bit unsigned integer          | `0`, `255`                 |
| `uint16`         | `uint16_t`             | 16-bit unsigned integer         | `0`, `65535`               |
| `uint32`         | `uint32_t`             | 32-bit unsigned integer         | `0`, `4000000000`        |
| `uint64`         | `uint64_t`             | 64-bit unsigned integer         | `0`, `18e18`              |
| `float32`        | `float`                | 32-bit floating point           | `1.0`, `-3.14`, `1.5e-8`   |
| `float64`        | `double`               | 64-bit floating point (default) | `1.0`, `-3.14`, `1.5e-8`   |
| `bool`           | `bool` (`<stdbool.h>`) | Boolean                         | `true`, `false`            |
| `rune`           | `char`                 | Single Unicode character        | `'a'`, `'\n'`, `'∑'`       |
| `string`         | `char*`                | Sequence of characters          | `"hello"`, `"\"quoted\""`   |
| `void`           | `void`                 | Represents absence of type      | (Function return only)     |

## Compound Types

*(Currently only primitive types are supported. Arrays, structs, etc., may be added later.)*

## Type Compatibility and Conversions

- **Strict Equality:** For most operations (assignment, binary ops), types must match exactly. For example, you cannot assign an `int32` to an `int64` or add an `int32` and a `float32` without explicit conversion (if/when conversion functions are added).
- **Comparisons (`eq`, `neq`, `lt`, etc.):** Allow comparison between any two operands of the *same* numeric type (`int*`, `uint*`, `float*`), the same `bool` type, or the same `rune` type. String comparison might be added later.
- **Logical Ops (`and`, `or`, `not`):** Require `bool` operands.
- **No Implicit Conversions:** The language currently does not support implicit type conversions (e.g., `int` to `float`).

## Variables

- **Declaration and Initialization:** Variables must be explicitly declared and initialized on the same line.
The `declare` keyword is **removed**.
- **Mutability:** Variables can be declared as either mutable or immutable.
    - **Immutable:** Declared using `let`. Requires initialization. Cannot be reassigned using `set`.
      ```monopressed
      let int32 count 10
      let bool is_done false
      let string message "Hello"
      # set count 20 # Error: Cannot assign to immutable variable 'count'
      ```
    - **Mutable:** Declared using `init`. Requires initialization. Can be reassigned using `set` with a value of the *same type*.
      ```monopressed
      init int32 score 0
      init float64 factor 1.0
      
      set score 100      # OK
      set factor 0.5     # OK
      # set score 1.5    # Error: Type mismatch
      ```
- **Scope:** Variables have lexical scope (global or function scope). (Details on shadowing TBD).
- **Unused Variables:** Declaring a variable without using it results in a semantic error.

## Functions

- **Definition:** Functions are defined using `def`, specifying parameter names and types, and an explicit return type (`void` if no value is returned).
  ```monopressed
  def add_numbers(int32 a, int32 b) int32
      return (add a b)

  def print_message(string msg) void
      print_string msg
      return
  ```
- **Return Types:** The type of the expression in a `return` statement must match the function's declared return type. An empty `return` is required for `void` functions and disallowed otherwise.
- **Function Calls:** Arguments passed to functions must match the number and types of the declared parameters.

## Type Checking Rules (Initial Draft)

The semantic analyzer will enforce the following rules (this list will expand):

1.  **Definition:** Variables must be defined using `let` (immutable) or `init` (mutable) before use within their scope. Re-definition within the same scope is an error.
2.  **Assignment:** The type of the expression being assigned (`set` or `init`) must exactly match the declared type of the variable. Assignment using `set` is only allowed for variables defined with `init`. No implicit conversions are allowed between different primitive types at this stage (e.g., assigning an `int32` to a `float32` variable is an error).
3.  **Operators:**
    *   Arithmetic (`add`, `sub`, `mul`, `div`, `fadd`, etc.): Operands must be of compatible numeric types and must match each other. Integer operations require integer types (`intX`, `uintX`). Floating-point operations (`fadd`, etc.) require floating-point types (`float32`, `float64`). The result type matches the operand types. Mixing integer and float in a single operation is an error without explicit casting (casting TBD).
    *   Comparison (`eq`, `neq`, `lt`, `lte`, `gt`, `gte`): Operands must be of the same comparable type (numeric, bool, rune, string). The result is always `bool`.
    *   Logical (`and`, `or`): Operands must be `bool`. The result is `bool`.
    *   Unary (`not`): Operand must be `bool`. The result is `bool`.
4.  **Function Calls:**
    *   The number of arguments provided must match the number of parameters defined.
    *   The type of each argument must exactly match the type of the corresponding parameter.
    *   Called functions must be defined.
5.  **Return Statements:** The type of the expression in a `return` statement must match the declared return type of the function. A `return` with no expression is only valid in functions declared with a `unit` return type.
6.  **Control Flow Conditions:** Expressions used as conditions (`if`, `elif`, `while`) must evaluate to `bool`.
7.  **Loop Control:** `break` and `continue` statements are only allowed inside `while` or `for` loops.

*(Note: This is a starting point. Rules for arrays, structs, and potential type conversions/casting will be added later.)* 