from functools import wraps
from enum import Enum, auto
import inspect
from tabnanny import check


class ConstraintOp(Enum):
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    def __repr__(self) -> str:
        if self is ConstraintOp.EQ:
            return '=='
        elif self is ConstraintOp.NEQ:
            return '!='
        elif self is ConstraintOp.LT:
            return '<'
        elif self is ConstraintOp.LE:
            return '<='
        elif self is ConstraintOp.GT:
            return '>'
        else:
            return '>='

    def __str__(self) -> str:
        return self.__repr__()


class ConstraintValue:
    def __eq__(self, other):
        return ConstraintExpr(self, other, ConstraintOp.EQ)

    def __neq__(self, other):
        return ConstraintExpr(self, other, ConstraintOp.NEQ)

    def __lt__(self, other):
        return ConstraintExpr(self, other, ConstraintOp.LT)

    def __le__(self, other):
        return ConstraintExpr(self, other, ConstraintOp.LE)

    def __gt__(self, other):
        return ConstraintExpr(self, other, ConstraintOp.GT)

    def __ge__(self, other):
        return ConstraintExpr(self, other, ConstraintOp.GE)


class ConstraintConst(ConstraintValue):
    def __init__(self, value) -> None:
        self.value = value

    def evaluate(self, _context):
        return self.value

    def __repr__(self) -> str:
        return f'{self.value}'


class ConstraintVariable(ConstraintValue):
    def __init__(self, var_name: str):
        self.name = var_name

    def evaluate(self, context_map):
        return context_map[self.name]

    def __repr__(self) -> str:
        return self.name


class ConstraintExpr:
    def __init__(self, lhs, rhs, op):
        if not isinstance(
            rhs, (ConstraintConst, ConstraintVariable, ConstraintExpr)
        ):
            self.rhs = ConstraintConst(rhs)
        else:
            self.rhs = rhs

        self.lhs = lhs
        self.op = op

    def evaluate(self, context_map):
        lhs = self.lhs.evaluate(context_map)
        rhs = self.rhs.evaluate(context_map)

        if self.op is ConstraintOp.EQ:
            return lhs == rhs
        elif self.op is ConstraintOp.NEQ:
            return lhs != rhs
        elif self.op is ConstraintOp.LT:
            return lhs < rhs
        elif self.op is ConstraintOp.LE:
            return lhs <= rhs
        elif self.op is ConstraintOp.GT:
            return lhs > rhs
        elif self.op is ConstraintOp.GE:
            return lhs >= rhs
        else:
            raise ValueError('operator must be ConstraintOp')

    def __repr__(self) -> str:
        return f'{self.lhs} {self.op} {self.rhs}'


def constraint(expr: ConstraintExpr):
    """
    Defines a constraint for the annotated function. Constraints are checked
    during function invocation. The annotated function will raise a ValueError
    if the constraints are not satisfied.
    """

    def decorator(func):
        first_time = False
        sig = inspect.signature(func)
        if not hasattr(func, 'constraints'):
            first_time = True
            func.constraints = list()

        func.constraints.append(expr)

        # avoid checking the constraints more times than necessary
        if not first_time:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_sig = sig.bind(*args, **kwargs)
            failed_constraints = []

            for constraint in func.constraints:
                if not constraint.evaluate(bound_sig.arguments):
                    failed_constraints.append(constraint)

            if failed_constraints:
                err_str = f'{failed_constraints[0]}'

                if len(failed_constraints) > 1:
                    for constraint in failed_constraints[1:]:
                        # make pretty error message
                        err_str += f',\n{" "*39}{constraint}'

                raise ValueError(f'unsatisfied constraint(s): {err_str}')
            return func(*args, **kwargs)

        return wrapper

    return decorator


CV = ConstraintVariable
CC = ConstraintConst
