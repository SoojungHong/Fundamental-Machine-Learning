import numpy as np

"""
Create an array of evenly spaced values (number of samples)
"""
np.linspace(0,2,9)
#print(np.linspace(0,2,9))  # between 0 and 2, create 9 numbers with evenly spaced samples


"""
Return a sample (or samples) from the “standard normal” distribution.
"""
# np.random.randn(x) : # x number of random value
# np.random.randn(a, b) : # shape is a x b

X = np.linspace(1, 10, 10)
np.random.randn(X.shape[0])
t= X.shape[0] # t = 10
print(np.random.rand(1, 2))


"""
d = np.arange(10,25,5) Create an array with evenly spaced value (step value)
This function does not indicate how many values should be created
"""
d = np.arange(10, 20, 3)

#print(d)


"""
np.random.randn() : return narray 

np.random.randn(n, m) : return narray with n x m shape 

n numbers of m length narray 

"""
x = np.random.randn(5,2)
print(np.random.randn(5,2))  # return list of list, each element list has two value, # of included lists is 5
#print(type(x[0]))  # add 10 to all
#print(x)
#print(x[0][1])

"""
np zero : create an array of zero 
"""
n_features = 10
print(np.zeros((1,n_features)) ) # row is one and n_features column

#print(np.zeros((3,4)) ) # number of row = 3, number of column = 4


# matrix
row = 3
col = 4

mat = [ [0] * col for _ in range(row) ]
#print(mat)
#print(mat[0][1])


"""
concatenate()
axis = 0 is default, it increase vertically, ie. increase row
axis = 1 is concatenate for each row's horizontally

"""
x1 = np.random.randn(5,2) + 5
x2 = np.random.randn(5,2) - 5
horizontal_concat_X = np.concatenate([x1, x2])
vertical_concat_X = np.concatenate([x1, x2], axis=0)

#print(np.concatenate(x1, x2))   # ERROR : concatenate should have [ ] outside

x1 = np.random.randn(5, 2) + 5
x2 = np.random.randn(5, 2) - 5
X = np.concatenate([x1, x2], axis=0)
theta = np.random.rand(X.shape[1])  # X.shape[1] is 2
print('X : ', X)
print('theta : ', theta)


"""
np.dot() : Dot product of two arrays. 
"""

tmp = np.dot(X, theta) # X's shape : (10, 2), theta shape is 2 --> dot product is (10, 1) i.e. 1 dimensional array with 10 element
#print('dot product:', tmp)



# random.binomial(n, p, size=None) : draw sample from binomial distribution
# n trials, p probability of success, size is output shape
dense = np.random.binomial(n=1, p=0.1, size=(10, 10))   # 1이 나올 확률
#print('dense : \n', dense)


# rng : random generator
from numpy.random import default_rng

rng = default_rng()
print(rng.permutation(5)) # permutation : Randomly permute a sequence, or return a permuted range.


"""
np.flatnonzero 

numpy.flatnonzero(a)

Return indices that are non-zero in the flattened version of a.
"""
tmp = [0, 0, 0, 0, 0, 1]
non_zero_idx = np.flatnonzero(tmp) # 5
print('flatten : ', non_zero_idx.flatten())

#inlier_ids = ids[self.n:][np.flatnonzero(thresholded).flatten()]


"""
np.hstack

Stack arrays in sequence horizontally (column wise).
"""
a = [1, 2, 3]
b = [4, 5, 6]
test = np.hstack([a, b])
print(test)

#if inlier_ids.size > self.d:
#    inlier_points = np.hstack([maybe_inliers, inlier_ids])



"""
np.linalg.inv 

Compute the (multiplicative) inverse of a matrix.

Given a square matrix a, return the matrix ainv satisfying dot(a, ainv) = dot(ainv, a) = eye(a.shape[0]). 
"""
#self.params = np.linalg.inv(X.T @ X) @ X.T @ y # TODO


"""
numpy eye() is 
2-D array with ones on the diagonal and zeros elsewhere.

"""



"""
np.linspace 
"""
line = np.linspace(-1, 1, num=100).reshape(-1, 1)


