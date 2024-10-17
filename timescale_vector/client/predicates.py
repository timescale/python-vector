import json
from datetime import datetime
from typing import Any, Literal, Union


class Predicates:
    logical_operators: dict[str, str] = {
        "AND": "AND",
        "OR": "OR",
        "NOT": "NOT",
    }

    operators_mapping: dict[str, str] = {
        "=": "=",
        "==": "=",
        ">=": ">=",
        ">": ">",
        "<=": "<=",
        "<": "<",
        "!=": "<>",
        "@>": "@>",  # array contains
    }

    PredicateValue = str | int | float | datetime | list | tuple  # type: ignore

    def __init__(
        self,
        *clauses: Union[
            "Predicates",
            tuple[str, PredicateValue],
            tuple[str, str, PredicateValue],
            str,
            PredicateValue,
        ],
        operator: Literal["AND", "OR", "NOT"] = "AND",
    ):
        """
        Predicates class defines predicates on the object metadata.
        Predicates can be combined using logical operators (&, |, and ~).

        Parameters
        ----------
        clauses
            Predicate clauses. Can be either another Predicates object
            or a tuple of the form (field, operator, value) or (field, value).
        Operator
            Logical operator to use when combining the clauses.
            Can be one of 'AND', 'OR', 'NOT'. Defaults to 'AND'.
        """
        if operator not in self.logical_operators:
            raise ValueError(f"invalid operator: {operator}")
        self.operator: str = operator
        if isinstance(clauses[0], str):
            if len(clauses) != 3 or not (isinstance(clauses[1], str) and isinstance(clauses[2], self.PredicateValue)):
                raise ValueError(f"Invalid clause format: {clauses}")
            self.clauses: list[
                Predicates
                | tuple[str, Predicates.PredicateValue]
                | tuple[str, str, Predicates.PredicateValue]
                | str
                | Predicates.PredicateValue
            ] = [clauses]
        else:
            self.clauses = list(clauses)

    def add_clause(
        self,
        *clause: Union[
            "Predicates",
            tuple[str, PredicateValue],
            tuple[str, str, PredicateValue],
            str,
            PredicateValue,
        ],
    ) -> None:
        """
        Add a clause to the predicates object.

        Parameters
        ----------
        clause: 'Predicates' or Tuple[str, str] or Tuple[str, str, str]
            Predicate clause. Can be either another Predicates object or a tuple of the form (field, operator, value)
            or (field, value).
        """
        if isinstance(clause[0], str):
            if len(clause) != 3 or not (isinstance(clause[1], str) and isinstance(clause[2], self.PredicateValue)):
                raise ValueError(f"Invalid clause format: {clause}")
            self.clauses.append(clause)
        else:
            self.clauses.extend(list(clause))

    def __and__(self, other: "Predicates") -> "Predicates":
        new_predicates = Predicates(self, other, operator="AND")
        return new_predicates

    def __or__(self, other: "Predicates") -> "Predicates":
        new_predicates = Predicates(self, other, operator="OR")
        return new_predicates

    def __invert__(self) -> "Predicates":
        new_predicates = Predicates(self, operator="NOT")
        return new_predicates

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Predicates):
            return False

        return self.operator == other.operator and self.clauses == other.clauses

    def __repr__(self) -> str:
        if self.operator:
            return f"{self.operator}({', '.join(repr(clause) for clause in self.clauses)})"
        else:
            return repr(self.clauses)

    def build_query(self, params: list[Any]) -> tuple[str, list[Any]]:
        """
        Build the SQL query string and parameters for the predicates object.
        """
        if not self.clauses:
            return "", []

        where_conditions: list[str] = []

        for clause in self.clauses:
            if isinstance(clause, Predicates):
                child_where_clause, params = clause.build_query(params)
                where_conditions.append(f"({child_where_clause})")
            elif isinstance(clause, tuple):
                if len(clause) == 2:
                    field, value = clause
                    operator = "="  # Default operator
                elif len(clause) == 3:
                    field, operator, value = clause
                    if operator not in self.operators_mapping:
                        raise ValueError(f"Invalid operator: {operator}")
                    operator = self.operators_mapping[operator]
                else:
                    raise ValueError("Invalid clause format")

                index = len(params) + 1
                param_name = f"${index}"

                if field == "__uuid_timestamp":
                    # convert str to timestamp in the database, it's better at it than python
                    if isinstance(value, str):
                        where_conditions.append(f"uuid_timestamp(id) {operator} ({param_name}::text)::timestamptz")
                    else:
                        where_conditions.append(f"uuid_timestamp(id) {operator} {param_name}")
                    params.append(value)

                elif operator == "@>" and isinstance(value, list | tuple):
                    if len(value) == 0:
                        raise ValueError("Invalid value. Empty lists and empty tuples are not supported.")
                    json_value = json.dumps(value)
                    where_conditions.append(f"metadata @> jsonb_build_object('{field}', {param_name}::jsonb)")
                    params.append(json_value)

                else:
                    field_cast = ""
                    if isinstance(value, int):
                        field_cast = "::int"
                    elif isinstance(value, float):
                        field_cast = "::numeric"
                    elif isinstance(value, datetime):
                        field_cast = "::timestamptz"
                    where_conditions.append(f"(metadata->>'{field}'){field_cast} {operator} {param_name}")
                    params.append(value)

        if self.operator == "NOT":
            or_clauses = " OR ".join(where_conditions)
            # use IS DISTINCT FROM to treat all-null clauses as False and pass the filter
            where_clause = f"TRUE IS DISTINCT FROM ({or_clauses})"
        else:
            where_clause = (" " + self.operator + " ").join(where_conditions)
        return where_clause, params
