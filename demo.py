from typing import Any, List
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

from interface import IntrinsicInterface, Resource, consumes


EthosUInterface = IntrinsicInterface('EthosUInterface')


@IntrinsicInterface.create_interface
class BadIdeaInterface:
    resources = {'test_resource': 1}


@IntrinsicInterface.create_interface
class BadIdeaInterface2:
    resources = [('test_resource2', 4)]


@consumes('test_resource2')
@consumes('test_resource')
@BadIdeaInterface.function
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


def gen_elementwise(a_shape, b_shape):
    pass


if __name__ == '__main__':
    print(EthosUInterface.registry)
    print(BadIdeaInterface.resources)
    print(BadIdeaInterface2.resources)
    print(test_fn.consumes)
