
import numpy as _np
import mxnet as mx
from mxnet import np, npx, autograd


# @with_seed()
# @use_np
def test_np_nan_to_num():
    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, _np.bool, _np.bool_,
              'int8', 'int32', 'float16', 'float32', 'float64', 'bool', None]
    
    print("nan_to_num")
    x = np.array([1,2,3,np.nan,np.inf,-np.inf])
    print("np.nan_to_num(x)",np.nan_to_num(x))
    x = np.array([1,2,3,np.nan,np.inf,-np.inf])
    print("np.nan_to_num(x,True,0,1000,-1000)",np.nan_to_num(x,True,0,1000,-1000))
    # objects = [
    #     [1,1]
    # ]
    # for dtype in dtypes:
    #     for src in objects:
    #         mx_arr = np.array(src, dtype=dtype)
    #         assert mx_arr.ctx == mx.current_context()
    #         if isinstance(src, mx.nd.NDArray):
    #             np_arr = _np.array(src.asnumpy(), dtype=dtype if dtype is not None else _np.float32)
    #         else:
    #             np_arr = _np.array(src, dtype=dtype if dtype is not None else _np.float32)
    #         assert type(mx_arr.dtype) == type(np_arr.dtype)


if __name__ == '__main__':
    test_np_nan_to_num()

