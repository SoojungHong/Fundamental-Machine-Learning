import numpy as np
from scipy.sparse import csr_matrix


# random.binomial(n, p, size=None) : draw sample from binomial distribution
# n trials, p probability of success, size is output shape
dense = np.random.binomial(n=1, p=0.1, size=(10, 10))   # 1이 나올 확률
print('dense : \n', dense)


# convert this matrix to sparse format
# csr_matrix : compressed sparse row matrix
sparse = csr_matrix(dense)
print('csr_matrix : \n', csr_matrix) # this is object type, therefore, not directly shows values


# create random matrix
random_mat = np.random.random(size=(10, 1))
print('random_mat : \n', random_mat)


# matrix dot operation (multiplication)
print('np dot : \n', np.dot(sparse, random_mat))

# Because np.dot doesn't natively recognize scipy.sparse matrices

print('np @ : \n', sparse @ random_mat)
