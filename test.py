from mxnet import np
a = np.arange(9).reshape(3,3)
b = np.arange(9).reshape(3,3).astype(np.int32)
c = np.full_like(a, 1)
print(a)
print(b)
print(c)