from collections import namedtuple
from dataclasses import dataclass, field
import inspect
import dis
from functools import wraps, partial

from tvm.tir.function import PrimFunc
from tvm.script import tir as T, from_source
from tvm import tir
from tvm.script.diagnostics import TVMDiagnosticCtx
from typing import Any, Callable, Dict, Iterable, List, Tuple

from constraints import constraint, CV, CC

Resource = namedtuple('Resource', ['name', 'count'])


@dataclass(frozen=True)
class IntrinsicDeclaration:
    desc: PrimFunc
    impl: PrimFunc
    name: str
    consumes: List[Resource] = field(default_factory=list)


def consumes(resource_name, count=1):
    def decorator(func):
        if hasattr(func, 'consumes') and isinstance(func.consumes, list):
            func.consumes.append(Resource(resource_name, count))
        elif hasattr(func, 'consumes'):
            raise TypeError(
                f'Consumes decorator expects the consumes attribute to be a list, but got {type(func.consumes)}'
            )
        else:
            func.consumes = list()
            func.consumes.append(Resource(resource_name, count))
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

    if hasattr(func, 'consumes'):
        inner.consumes.extend(func.consumes)

    return inner


class IntrinsicInterface:
    registry: Dict[str, IntrinsicDeclaration]
    name: str
    resources: Dict[str, int]

    def __init__(self, name):
        self.registry = dict()
        self.name = name
        self.resources = dict()

    def function(self, func=None, *, name=None):
        if func is None:
            return partial(self.function, name=name)

        name = name or f'{self.name}_{func.__name__}'

        inner = function(func, name=name)
        self.registry[name] = inner

        return inner

    def add_resource(self, name: str, count: int):
        self.resources[name] = count

    @staticmethod
    def create_interface(cls):
        """
        Utility decorator used to turn a class declaration into an interface
        object. It's probably better to just make the object directly by
        initializing an IntrinsicInterface
        """

        inner = IntrinsicInterface(cls.__name__)
        if hasattr(cls, 'resources'):
            # the validation code in these if-arms is inefficient but given that
            # these resource dicts are unlikely to be all that large this is
            # fine for the time being. If the need arises these can be converted
            # to a more efficient but less readable single pass version

            if isinstance(cls.resources, dict) and all(
                (
                    isinstance(key, str) and isinstance(value, int)
                    for key, value in cls.resources.items()
                )
            ):
                inner.resources = cls.resources
            elif isinstance(cls.resources, Iterable) and all(
                (
                    isinstance(item, (Resource, tuple)) and len(item) == 2
                    for item in cls.resources
                )
            ):
                inner.resources = {key: value for key, value in cls.resources}
                cls.resources = inner.resources
            else:
                raise TypeError(
                    'the resources field must be a dictionary with string keys and int values or an Iterable of valid Resource tuples'
                )
        else:
            cls.resources = inner.resources

        class WrappedIntrinsicInterface(cls):
            _inner = inner
            registry = _inner.registry

            @classmethod
            def function(cls, *args, **kwargs):
                return cls._inner.function(*args, **kwargs)

        return WrappedIntrinsicInterface


# friendly alias
create_interface = IntrinsicInterface.create_interface


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
