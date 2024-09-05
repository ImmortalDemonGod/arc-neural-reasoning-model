from typing import List

def to_cnf(formula: List[str]) -> List[List[str]]:
    def parse_clause(clause: str) -> List[str]:
        return [lit.strip() for lit in clause.replace('(', '').replace(')', '').split('|')]

    def distribute_and(clauses: List[List[str]]) -> List[List[str]]:
        result = [[]]
        for clause in clauses:
            result = [r + [c] for r in result for c in clause]
        return result

    cnf = []
    for expr in formula:
        if '&' in expr:
            clauses = [parse_clause(clause) for clause in expr.split('&')]
            cnf.extend(distribute_and(clauses))
        else:
            cnf.append(parse_clause(expr))

    return cnf

def from_dnf(dnf: List[List[str]]) -> List[List[str]]:
    if not dnf:
        return []
    
    result = [[term] for term in dnf[0]]
    
    for clause in dnf[1:]:
        new_result = []
        for term in clause:
            for existing in result:
                new_result.append(existing + [term])
        result = new_result
    
    return result
