import pytest
from app.core.ast_normalizer import canonicalize_code
from app.core.code_engine import compare_python_code


def test_canonicalize_code_renaming_and_docstrings():
    code = """
def calculate_sum(numbers):
    '''Calculate total of items.'''
    # Local accumulator
    total = 0
    for n in numbers:
        total += n
    return total
"""
    canonical_source, node_seq = canonicalize_code(code)

    assert "calculate_sum" not in canonical_source
    assert "Calculate total of items." not in canonical_source
    assert "numbers" not in canonical_source
    assert "func_1" in canonical_source
    assert "var_1" in canonical_source
    assert "FunctionDef" in node_seq
    assert "For" in node_seq


def test_code_logic_match_variable_renaming_example():
    code_a = """
def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total
"""
    code_b = """
def find_total(data):
    result = 0
    for item in data:
        result += item
    return result
"""
    # Compare with UniXcoder
    res = compare_python_code(code_b, code_a, use_unixcoder=True)

    assert res["syntax_valid"] is True
    assert res["variable_renaming_detected"] is True
    assert res["structural_similarity"] >= 0.95
    assert res["semantic_similarity"] >= 0.85
    assert res["match_type"] == "Semantic Code Logic Match (High Confidence Refactor)"
    assert "evidence of shared logic patterns" in res["evidence_note"]


def test_unrelated_code_architectures():
    code_a = """
def bubble_sort(items):
    n = len(items)
    for i in range(n):
        for j in range(0, n - i - 1):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
    return items
"""
    code_b = """
def fetch_user_record(user_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
"""
    res = compare_python_code(code_b, code_a, use_unixcoder=False)

    assert res["syntax_valid"] is True
    assert res["variable_renaming_detected"] is False
    assert res["structural_similarity"] < 0.60
    assert res["match_type"] == "Distinct Implementation"


def test_syntax_error_handling():
    invalid_code = "def broken_function( incomplete_syntax"
    valid_code = "def valid_func(): pass"

    res = compare_python_code(invalid_code, valid_code)
    assert res["syntax_valid"] is False
    assert "Syntax Compilation Failure" in res["match_type"]