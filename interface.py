from collections import namedtuple
from dataclasses import dataclass, field
from functools import singledispatch, wraps, partial
import inspect
import itertools
from tvm.tir.function import PrimFunc
from tvm.script import tir as T
from tvm import tir
from typing import Dict, Iterable, List

from constraints import constraint, CV, CC

Resource = namedtuple('Resource', ['name', 'count'])


@dataclass(frozen=True)
class IntrinsicDeclaration:
    desc: PrimFunc = field(repr=False)
    impl: PrimFunc = field(repr=False)
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


def function(func=None, *, name=None, desc_name='desc', impl_name='impl'):

    # this is based on recipe 9.6 from python cookbook and is here to allow the
    # decorator to be used both with and without the call syntax
    if func is None:
        return partial(
            function, name=name, desc_name=desc_name, impl_name=impl_name
        )

    name = name or func.__name__

    desc, impl = None, None

    if isinstance(func, type):
        desc = getattr(func, desc_name)
        impl = getattr(func, impl_name)

    else:
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

    def function(self, func=None, *, name=None, name_prefix=True, **kwargs):

        if func is None:
            return partial(
                self.function,
                name=name,
                name_prefix=name_prefix,
                **kwargs,
            )

        # this is a crime against humanity
        frame_stack = inspect.stack()
        if (
            len(frame_stack) > 2
            and frame_stack[2].function == '__call__'
            and 'self' in frame_stack[2].frame.f_locals
            and isinstance(
                frame_stack[2].frame.f_locals['self'], GeneratorWrapper
            )
        ):
            args = frame_stack[2].frame.f_locals['args']
            kwargs = frame_stack[2].frame.f_locals['kwargs']
            if args or kwargs:
                name = name or (
                    func.__name__
                    + '_'
                    + '_'.join((str(arg) for arg in args))
                    + '_'.join((str(val) for val in kwargs.values()))
                )

        name = name or f'{self.name}_{func.__name__}'

        if not name.startswith(self.name) and name_prefix:
            name = f'{self.name}_{name}'

        assert (
            name not in self.registry
        ), f'An intrinsic named {name} has already been registered, a second intrinsic cannot be registered under the same name. If using a generator, please ensure that the generator produces a uniquely named intrinsic per each unique input set'

        inner = function(func, name=name, **kwargs)
        self.registry[name] = inner

        return inner

    def set_resource(self, name: str, count: int):
        self.resources[name] = count

    def set_resources_from_dict(self, resource_dict: dict):
        self.resources.update(
            (
                (key, value)
                for key, value in resource_dict.items()
                if isinstance(key, str) and isinstance(value, int)
            )
        )

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
            function = _inner.function

        return WrappedIntrinsicInterface


class GeneratorWrapper:
    """
    A wrapper class for functions which generate intrinsic implementations. Used
    to handle automatic name mangling based on generator input
    """

    def __init__(self, wrapped_fn):
        self._wrapped_fn = wrapped_fn

    def __call__(self, *args, **kwargs):
        self._wrapped_fn(*args, **kwargs)


def generator(func):
    """
    Decorator for wrapping functions which generate intrinsic implementations
    """
    gen = GeneratorWrapper(func)
    gen = wraps(func)(gen)
    return gen


def run_generator(**kwargs):
    """
    This decorator when given iterable keyword arguments will run the
    annotated generator with the cartesian product of the input iterators.

    Note: wraps the generator function with the @generator wrapper if it has not
    already been applied
    """

    product = itertools.product(*kwargs.values())

    def decorator(func):
        if not isinstance(func, GeneratorWrapper):
            func = generator(func)

        signature = inspect.signature(func)
        arg_ordering = [
            key
            for key, value in signature.parameters.items()
            if value.kind == value.POSITIONAL_ONLY
        ]

        for concrete_values in product:
            arg_dict = {
                key: value
                for key, value in zip(kwargs.keys(), concrete_values)
            }

            args = [arg_dict[key] for key in arg_ordering]
            arg_dict = {
                key: value
                for key, value in arg_dict.items()
                if key not in arg_ordering
            }
            bound = signature.bind(*args, **arg_dict)
            bound.apply_defaults()
            func(*bound.args, **bound.kwargs)

        return func

    return decorator


# friendly alias
create_interface = IntrinsicInterface.create_interface
