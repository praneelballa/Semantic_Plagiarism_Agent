import ast
from typing import Dict, Tuple
import astor


class PythonASTNormalizer(ast.NodeTransformer):
    """
    Traverses and canonicalizes Python Abstract Syntax Trees:
    - Normalizes local variable names to var_1, var_2, ...
    - Normalizes function definition names to func_1, func_2, ...
    - Preserves language keywords, built-ins, and control-flow structure.
    - Strips docstrings and comment blocks.
    """

    def __init__(self):
        super().__init__()
        self.var_map: Dict[str, str] = {}
        self.func_map: Dict[str, str] = {}
        self.var_counter: int = 0
        self.func_counter: int = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Strip function docstring if present
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)

        # Canonicalize function name (preserve dunder methods like __init__)
        if not (node.name.startswith("__") and node.name.endswith("__")):
            if node.name not in self.func_map:
                self.func_counter += 1
                self.func_map[node.name] = f"func_{self.func_counter}"
            node.name = self.func_map[node.name]

        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(node)  # type: ignore

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Ignore Python built-ins (e.g., range, len, print, sum)
        if node.id not in dir(__builtins__):
            if node.id not in self.var_map:
                self.var_counter += 1
                self.var_map[node.id] = f"var_{self.var_counter}"
            node.id = self.var_map[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        # Canonicalize function parameter arguments
        if node.arg != "self" and node.arg not in dir(__builtins__):
            if node.arg not in self.var_map:
                self.var_counter += 1
                self.var_map[node.arg] = f"var_{self.var_counter}"
            node.arg = self.var_map[node.arg]
        self.generic_visit(node)
        return node


def canonicalize_code(code_str: str) -> Tuple[str, str]:
    """
    Parses Python source code and returns:
    (canonicalized_source_code, structural_node_sequence)
    """
    if not code_str or not code_str.strip():
        return "", ""

    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        raise ValueError(f"Python syntax compilation failed: {exc}")

    # Strip module-level docstring
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        tree.body.pop(0)

    normalizer = PythonASTNormalizer()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)

    canonical_source = astor.to_source(normalized_tree).strip()
    node_sequence = "-".join(type(n).__name__ for n in ast.walk(normalized_tree))

    return canonical_source, node_sequence