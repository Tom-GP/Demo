import numpy as np

a = np.array([8, 2, 0, 10, 15])
b = np.arange(5)
c = np.array([[1, 2], [2, 3], [4, 5]])
print(a.itemsize)
print(c.dtype)
d = np.array([2, 4, 6, 8], dtype=np.float64)
print(d.itemsize)
print(d)
print(c)
print(c.shape)
print(np.zeros((3, 4), dtype=np.int64))