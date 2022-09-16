import numpy as np
import math


def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache


def batch_normalize(X):

    # learnable parameters : gamma, beta by backpropagation

    # The method to normalize the batch X :
    # X = (gamma * ((X - mean_x)/sqrt(variance_X**2 + epsilon))) + beta

    # init  # TODO : what value should be init value?
    gamma = 1
    epsilon = 0.01
    beta = 1

    # mean of X
    mean_X = np.mean(X)
    variance_X = np.var(X)

    normalized_X = (gamma * ((X - mean_X)/math.sqrt(variance_X**2 + epsilon))) + beta

    return normalized_X


def main():
    X = np.random.randn(10, 5)
    print(batch_normalize(X))


if __name__ == "__main__":
    main()