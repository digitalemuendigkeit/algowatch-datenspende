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
    centroids = np.random.choice(len(res_tmp), k, replace=False)
    centroids.sort()
    
    # initialize clusters
    clusters = np.repeat(None, len(res_tmp))
    counter = 0
    for c in centroids:
        clusters[c] = counter
        counter = counter + 1

    # initialize data frame
    df = pd.DataFrame(
        {
            'elem_index': [elem for elem in range(len(res_tmp))],
            'cluster': clusters,
            'mean_rbo': np.repeat(None, len(res_tmp))
        }
    )

    # k-means iterations
    for steps in range(n_iterations):
        
        # test print
        print(centroids)

        # create an index list and drop the indices of the centroids
        idx = [elem for elem in range(len(res_tmp))]
        drop_counter = 0
        for k_idx in centroids:
            idx.pop(k_idx - drop_counter)
            drop_counter = drop_counter + 1

        # iterate over index list
        for i in idx:
            rbo_dist = []
            for c in centroids:
                rbo_dist.append(rbo.rbo_ext(res_tmp[i], res_tmp[c], 0.9))
            df.iloc[i, 1] = rbo_dist.index(max(rbo_dist))
        centroids = update_centroids(df, res_tmp, k)
    
    # drop mean_rbo column
    df = df.drop(columns = 'mean_rbo')

    return df, centroids

#%%

def update_centroids(df, res, n_clust):
    """update centroids after k-means iteration"""
    
    # empty list for new centroids
    centroids_updated = np.repeat(None, n_clust)
    
    # iterate over clusters
    for k in df.cluster.unique():
        
        # subset data frame for cluster k
        cluster = df[df.cluster == k]
        
        # iterate over cluster k
        for i in cluster.elem_index:
            
            # initialize temporary list for mean rbo scores
            tmp = []
           
            # compute rbo to all other lists in the same cluster
            for j in cluster.elem_index:
                tmp.append(rbo.rbo_ext(res[i], res[j], 0.9))
            
            # save mean of rbo to other results in cluster
            cluster[cluster['elem_index'] == i][2] = np.mean(tmp)
        
        # update new centroids list
        centroids_updated[k] = cluster['elem_index'].idxmax(axis=0)
    
    # sort updated centroids
    centroids_updated.sort()

    return centroids_updated

#%%
