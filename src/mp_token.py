import logging
from dataclasses import dataclass
from typing import Any, Optional, Dict

TT_DEC = "dec"
TT_SELFSET = "selfset" # Added for self-setting operation
TT_RETURN = "return"
TT_RANGE = "range"
TT_AND = "and"

TT_WHILE = "while"
TT_FOR = "for"
TT_INC = "inc"
TT_MOD = "mod"
TT_ELSE = "else"
TT_IF = "if"
TT_BOOL = "bool"
TT_NULL = "null"
TT_SET = "set"
TT_GET = "get"
TT_ADD = "add"
TT_SUB = "sub"
TT_MUL = "mul"
TT_DIV = "div"
TT_OR = "or"
TT_NOT = "not"

KEYWORDS: Dict[str, str] = {
    "true": TT_BOOL,
    "false": TT_BOOL,
    "null": TT_NULL,
    "set": TT_SET,
    "get": TT_GET,
    "add": TT_ADD,
    "sub": TT_SUB,
    "mul": TT_MUL,
    "div": TT_DIV,
    "mod": TT_MOD,
    "if": TT_IF,
    "else": TT_ELSE,
    "while": TT_WHILE,
    "for": TT_FOR,
    "inc": TT_INC,
    "dec": TT_DEC,
    "selfset": TT_SELFSET,
    "return": TT_RETURN,
    "range": TT_RANGE,
    "and": TT_AND,
    "or": TT_OR,
    "not": TT_NOT,
} 