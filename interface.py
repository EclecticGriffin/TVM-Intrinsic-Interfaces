from collections import namedtuple
from dataclasses import dataclass
import inspect
import dis
from functools import wraps, partial

from tvm.tir.function import PrimFunc
from tvm.script import tir as T, from_source
from tvm import tir
from tvm.script.diagnostics import TVMDiagnosticCtx
from typing import Any, Callable, Dict, Tuple

from constraints import constraint, CV, CC

# def extract_impl_desc(input_mod: ast.Module, impl_name, desc_name):
#     assert len(input_mod.funcs) == 1
#     tir_prefix = ['T', 'tir']

#     _name, outer_fn = list(input_mod.funcs.items())[0]

#     impl, desc = None, None

#     for stmt in outer_fn.body.stmts:
#         if isinstance(stmt, ast.Function):
#             if stmt.name == impl_name:
#                 impl = stmt
#             elif stmt.name == desc_name:
#                 desc = stmt

#     desc = from_source(str(desc))
#     impl = from_source(str(impl))

#     return


@dataclass
class IntrinsicDeclaration:
    desc: PrimFunc
    impl: PrimFunc
    name: str


Resource = namedtuple('Resource', ['name', 'count'])


def consumes(resource_name, count=1):
    def decorator(func):
        if hasattr(func, 'consumes'):
            func.consumes.add(Resource(resource_name, count))
        else:
            func.consumes = set()
            func.consumes.add(Resource(resource_name, count))
        return func

    return decorator


def function(func=None, *, name=None):

    # this is based on recipe 9.6 from python cookbook and is here to allow the
    # decorator to be used both with and without the call syntax
    if func is None:
        return partial(function, name=name)

    name = name or func.__name__
    desc, impl = func()

    tir.TensorIntrin.register(name, desc, impl)
    inner = IntrinsicDeclaration(desc, impl, name)
    return inner


class IntrinsicInterface:
    registry: Dict[str, IntrinsicDeclaration]

    def __init__(self, name):
        self.registry = dict()
        self.name = name

    def function(self, func=None, *, name=None):
        if func is None:
            return partial(self.function, name=name)

        name = name or f'{self.name}_{func.__name__}'

        inner = function(func, name=name)
        self.registry[name] = inner

        return inner

    @staticmethod
    def create_interface(cls):
        """
        Utility decorator used to turn a class declaration into an interface
        object. It's probably better to just make the object directly by
        initializing an IntrinsicInterface
        """

        class WrappedIntrinsicInterface(cls):
            _inner = IntrinsicInterface(cls.__name__)
            registry = _inner.registry

            @classmethod
            def function(cls, *args, **kwargs):
                return cls._inner.function(*args, **kwargs)

        return WrappedIntrinsicInterface


@consumes('test resource', 5)
@function
def test_mma_intrin() -> Tuple[Callable, Callable]:
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
        B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
        C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

        with T.block('root'):
            T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            for i, j, k in T.grid(16, 16, 16):
                with T.block('update'):
                    vii, vjj, vkk = T.axis.remap('SSR', [i, j, k])
                    C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]

    @T.prim_func
    def impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
        B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
        C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

        with T.block('root'):
            T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            T.evaluate(
                T.tvm_mma_sync(
                    C.data,
                    C.elem_offset // 256,
                    A.data,
                    A.elem_offset // 256,
                    B.data,
                    B.elem_offset // 256,
                    C.data,
                    C.elem_offset // 256,
                    dtype='handle',
                )
            )

    return desc, impl


# @constraint(CV('A') < 5)
# @constraint(CV('B') <= 5)
# # @constraint(CC(0) < CV('A') < 7)
# def test_fn(A, B):
#     pass


# test_fn(4, 5)


# test_fn(5, 6)
