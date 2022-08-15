from tvm.script import tir as T

from interface_extension.interface import (
    IntrinsicInterface,
    Resource,
    consumes,
    run_generator,
    generator,
)
from interface_extension.constraints import constraint, CV, CC

TestInterface = IntrinsicInterface('TestInterface')
TestInterface.set_resources_from_dict(
    {
        'resource_A': 5,
        'resource_B': 1,
        'other_resource': 2,
        'yet another resource': 12,
        'test_resource': 2,
        'test_resource2': 2,
    }
)


@IntrinsicInterface.create_interface
class SecondInterface:
    resources = {'test_resource': 1}


@consumes('test_resource2')
@consumes('test_resource')
@TestInterface.function
def test_fn():
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


@consumes('test_resource2')
@consumes('test_resource')
@TestInterface.function
class alt_fn:
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


def v1(a_shape, b_shape=None):
    return a_shape < b_shape


def v2(a_shape):
    return a_shape % 2


@run_generator(
    a_shape=[1, 2, 3], b_shape=(x for x in range(0, 10)), _validators=[v1, v2]
)
def gen_elementwise(a_shape, b_shape):
    @TestInterface.function
    class generated:
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


@constraint(CV('a') < 10)
@constraint(0 < CV('c'))
@constraint(10 <= CV('b'))
def constraint_example(a, b, c):
    pass


if __name__ == '__main__':
    import pprint

    gen_elementwise(5, 124)

    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(TestInterface.registry)
