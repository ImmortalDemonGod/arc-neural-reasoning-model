import pytest
from ...src.sat_solver.cnf_converter import to_cnf, from_dnf

def test_to_cnf_simple():
    assert to_cnf(['A', 'B']) == [['A'], ['B']]

def test_to_cnf_or():
    assert to_cnf(['A | B']) == [['A', 'B']]

def test_to_cnf_and():
    assert to_cnf(['A & B']) == [['A'], ['B']]

def test_to_cnf_complex():
    assert to_cnf(['(A | B) & (C | D)']) == [['A', 'B'], ['C', 'D']]

def test_to_cnf_negation():
    assert to_cnf(['~(A & B)']) == [['~A', '~B']]

def test_from_dnf_simple():
    assert from_dnf([['A'], ['B']]) == [['A', 'B']]

def test_from_dnf_complex():
    assert from_dnf([['A', 'B'], ['C', 'D']]) == [['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D']]

def test_from_dnf_with_negation():
    assert from_dnf([['~A', 'B'], ['C']]) == [['~A', 'C'], ['B', 'C']]
