import numpy as np
import random, collections

"""
find the miminum distance centroid 
"""
def kmeans_predict(mus, x): # mus : mean of smallest distance??
    # find mean with smallest dist from X
    # if two mus are equidistance from X, returns the first one it checked
    min_dist = np.Inf   # value is inf
    best_mu = np.Inf    # value is inf

    for mu in mus:
        dist = np.linalg.norm(x-mu)
        if dist < min_dist:
            min_dist = dist
            best_mu = mu
    return best_mu

"""
if previous centroid (pre_mus) is equal to current centroid (mus) 
"""
def has_converged(mus, pre_mus):
    for x, y in zip(mus, pre_mus):  # zip() : The zip() function takes iterables (can be zero or more), aggregates them in a tuple, and returns it.
        #print('x, y : ', x, y)
        if (x == y).all():  # The all() function returns True if all elements in the given iterable are true. If not, it returns False.
            return True
    else:
        return False


"""
Assign all points in X to mus 

by assigning each point x to centroids, forming the clusters
"""
def _cluster(X, mus):
    clusters = collections.defaultdict(list)
    for x in X:
        #print('x: ', x) # given point x_i, measure the distance to all centroids (in mus), e.g two centroids
        dists = [np.linalg.norm(x-mu) for mu in mus]    # norm() : return one of eight different matrix norms
        #print('dists: ', dists)
        min_idx = dists.index(min(dists))   # The index() returns the index of the specified element in the list.

        # 존재하는 클러스터 중 가장 가까운 클러스터에 할당한다.
        clusters[min_idx].append(x) # there are (e.g.) two centroids and we know their index, put points that belongs to each centroid
        #print('clusters : ', clusters)
    return clusters


def calc_centers(mus, clusters):
    mus = []
    for _, points in clusters.items():
        #print('_ : ', _)    # TODO : _ ?? --> value is 0 and 1 for each iteration
        #print('points : ', points)  # points are several points that belong to indexed' centroid

        mus.append(np.mean(points, axis=0)) # calculate the mean of each group (points), it is new centroids
    return mus


"""

train
1. set initial centroids with number of cluster k 
2. while not converged and smaller iterations, with given data points and with initial centroids, 
3. find clusters (i.e. find points that belongs to each centroid) 
4. with newly formed clusters, calculate new centroid
5. do 2 ~ 4 until converge or pre-set iterations

test/prediction 
given centroids with training, find the closest centroid 

"""
def kmeans_fit(X, k):
    # Initialize to K random centers

    # mus = X[np.random.choice(X.shape[0], k, False)] #ORG
    t = np.random.choice(X.shape[0], k, False)  # choice() : Generates a random sample from a given 1-D array , k개의 값 (실은 인덱스) 초이스
    mus = X[t] # get t indexed tuples, e.g. t = [0 5]
    #print('t : ', t)
    #print('mus : ', X[t])   # [[-6.28603202 -2.68989861] [ 4.14454348  4.19630792]]

    pre_mus = mus + 2   # because to initially set pre_mus differently than mus, then while below can proceed until converge
    #print('pre_mus : ', pre_mus)
    max_it, it = 10, 1

    while it < max_it and not has_converged(mus, pre_mus):
        pre_mus = mus   # set current mus to pre_mus
        #print('pre_mus : ', pre_mus)

        # Assign all points in X to clusters
        clusters = _cluster(X, mus)

        # Reevaluate centers
        mus = calc_centers(mus, clusters)
        it += 1

    return (mus, clusters)


"""
 main function
 - train kmeans algorithm 
 - predict with trained kmeans algorithm 
"""
def main():
    x1 = np.random.randn(5, 2) + 5  # randn() : Return a sample (or samples) from the “standard normal” distribution. 5 by 2 array
    x2 = np.random.randn(5, 2) - 5
    X = np.concatenate([x1, x2], axis=0)

    #print('x1 : ', x1)
    #print('x2 : ', x2)
    #print('X : ', X)

    k = 2
    mus, clusters = kmeans_fit(X, k)
    print(mus, '\n')  # centroids
    print(clusters)   # points for each clusters
    print('best_mu : ', kmeans_predict(mus, np.array([6.02, 6.656])))


# call main function
if __name__ == "__main__":
    main()
