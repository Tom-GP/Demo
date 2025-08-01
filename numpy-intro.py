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
print(np.ones((5, 6)))
e = np.linspace(1, 9, 10)
print(e)
print(np.array([[1, 2, 3], [6, 5, 4]]).reshape(3, 2))
print(np.array([[1, 2, 3], [6, 5, 4]]).ravel())
print(c.sum(axis=1))

f = np.array([[2, 4, 8], [5, 7, 1], [1, 2, 3]])
print(f[:2, :2])
for row in f:
    print(row)
for cell in f.flat:
    print(cell)
print(np.vstack((c.reshape(2, 3),f)))
# print(np.hstack((c, f.reshape(5, 2))))
print(np.hsplit(np.arange(30).reshape(3, 10), 5)) # spliting horizontally into 5 different sizes arrays

h = np.arange(15).reshape(3, 5)
g = h>7
print(h)
print(g)
h[g] = 0
print(h)