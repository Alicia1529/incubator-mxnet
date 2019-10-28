
import os
import unittest
import numpy as _np
import mxnet as mx
from mxnet import np, npx, autograd
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray, retry, use_np
from common import with_seed, TemporaryDirectory
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf, assert_exception, is_op_runnable
from mxnet.ndarray.ndarray import py_slice
from mxnet.base import integer_types
import scipy.stats as ss


# @with_seed()
# @use_np
# def test_np_nan_to_num():
#     dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, _np.bool, _np.bool_,
#               'int8', 'int32', 'float16', 'float32', 'float64', 'bool', None]
    
#     print("nan_to_num")
#     x = np.array([1,2,3,np.nan,np.inf,-np.inf])
#     print("np.nan_to_num(x)",np.nan_to_num(x))
#     print("x:",x)
#     x = np.array([1,2,3,np.nan,np.inf,-np.inf])
#     print("np.nan_to_num(x,True,0,1000,-1000)",np.nan_to_num(x,True,0,1000,-1000))
#     print("x:",x)
#     objects = [
#         # 0,
#         # 1,
#         [1,2,3,4],
#         [0,1.1,2.2,3.3]
#     ]
#     dic = {"nan":[0, 0.0, 1, 1.0, 1.1], "inf":[0,0.0,1000,1000.0,1.7976931348623157e+308],"-inf":[0,0.0,-1000,-1000.0,-1.7976931348623157e+308]}

#     rand_ndarray(1).as_np_ndarray()

    # data = mx.symbol.Variable('data')

    # for dtype in dtypes:
    #     for src in objects:
    #         mx_arr = np.array(src, dtype=dtype)
    #         assert mx_arr.ctx == mx.current_context()
    #         np_arr =  _np.array(src, dtype=dtype if dtype is not None else _np.float32)
    #         assert np.nan_to_num(mx_arr) == _np.nan_to_num(np_arr)
            # for idx in range(5):
            #     assert np.nan_to_num(mx_arr, True, dic["nan"][idx], dic["inf"][idx], dic["-inf"][idx]) == _np.nan_to_num(np_arr, True, dic["nan"][idx], dic["inf"][idx], dic["-inf"][idx])

@with_seed()
@use_np
def test_np_nan_to_num():
    class TestNanToNum(HybridBlock):
        def __init__(self, copy, nan, posinf, neginf):
            super(TestNanToNum, self).__init__()
            self.copy = copy
            self.nan = nan
            self.posinf = posinf
            self.neginf = neginf
            # necessary initializations
            
        def hybrid_forward(self, F, a):
            return F.np.nan_to_num(a, self.copy, self.nan, self.posinf, self.neginf)
    
    objects = [
        -1,
        0,
        1,
        [-1, 0, 1],
        [[-1, 0, 1], [-1, 0, 1]]
    ]

    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, _np.bool, _np.bool_,
              'int8', 'int32', 'float16', 'float32', 'float64', 'bool', None]
    dic = {"nan":[0, 0.0, 1, 1.0, 1.1], "inf":[0,0.0,1000,1000.0,1.7976931348623157e+308],"-inf":[0,0.0,-1000,-1000.0,-1.7976931348623157e+308]}
    
    [atol, rtol] = [1e-6, 1e-5]

    ctx = ctx if ctx else default_context()
    for hybridize in [True, False]:
        for src in objects:
            for dtype in dtypes:
                for idx in range(5):
                    for copy in [True, False]:
                        test_np_nan_to_num = TestNanToNum(copy, dic["nan"][idx], dic["inf"][idx], dic["-inf"][idx])
                        if hybridize:
                            test_np_nan_to_num.hybridize()
                        x1 = mx.nd.array(src, dtype=dtype, ctx=ctx).asnumpy()/0.0
                        x2 = mx.nd.array(src, dtype=dtype, ctx=ctx)/0.0
                        np_out = _np.nan_to_num(x1, copy, dic["nan"][idx], dic["inf"][idx], dic["-inf"][idx])
                        mx_out = test_np_nan_to_num(x2, copy, dic["nan"][idx], dic["inf"][idx], dic["-inf"][idx])
                        assert mx_out.shape == np_out.shape
                        assert mx_out.dtype == np_out.dtype
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
                        if copy == False:
                            assert x1.shape == x2.shape
                            assert x1.dtype == x2.dtype
                            assert_almost_equal(x1, x2, rtol=rtol, atol=atol)                          

if __name__ == '__main__':
    test_np_nan_to_num()

