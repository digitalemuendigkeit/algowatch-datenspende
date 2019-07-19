#%%
import rbo
import numpy as np
import pandas as pd

#%%

def k_means_rbo(lol, k, n_iterations):
    """k-means algorithm adjusted for list clustering"""

    # preparation
    res_tmp = lol

    # choose k centroids at random (initialization)
    centroids = np.random.randint(0, len(res_tmp), k)

    # create an index list and drop the indices of the centroids
    idx = [elem for elem in range(len(res_tmp))]
    for k_idx in centroids:
        idx.pop(k_idx)
    
    # initialize clusters
    clusters = np.repeat(None, len(res_tmp))
    counter = 0
    for c in centroids:
        clusters[c] = counter
        counter = counter + 1

    df = pd.DataFrame(
        {
            'elem_index': [elem for elem in range(len(lol))],
            'cluster': clusters,
            'mean_rbo': np.repeat(None, len(res_tmp))
        }
    )

    # iterate
    for steps in range(n_iterations):
        for i in idx:
            rbo_dist = []
            for c in centroids:
                rbo_dist.append(rbo.rbo_ext(res_tmp[i], res_tmp[c], 0.9))
            df.iat[i, 1] = rbo_dist.index(min(rbo_dist))
        centroids = update_centroids(df, res_tmp, k)
    

    return df, centroids


def update_centroids(df, res, n_clust):
    """update centroids after k-means iteration"""
    # empty list for new centroids
    centroids = np.repeat(None, n_clust)
    # iterate over clusters
    for k in df.cluster.unique():
        # select lists for cluster
        cluster = df[df.cluster == k]
        for i in cluster.elem_index:
            tmp = []
            # compute rbo to all other lists in the same cluster
            for j in cluster.elem_index:
                tmp.append(rbo.rbo_ext(res[i], res[j], 0.9))
            # save mean of rbo to other results in cluster
            cluster.iat[i, 2] = np.mean(tmp)
        centroids[k] = cluster.idxmin(axis=0)[2]
    return centroids
            
                


    







#%%
