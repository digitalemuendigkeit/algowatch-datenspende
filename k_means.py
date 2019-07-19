#%%
import rbo
import numpy as np
import pandas as pd

#%%

def k_means_rbo(lol, k, n_iterations):
    """k-means algorithm adjusted for list clustering"""

    res_tmp = lol

    # choose k centroids
    centroids = np.random.randint(0, len(res_tmp), k)

    # create an index list and drop the indices of the centroids
    idx = [elem for elem in range(len(res_tmp))]
    for k_idx in centroids:
        idx.pop(k_idx)
    
    # initialize clusters
    clusters = np.repeat('NA', len(res_tmp))
    counter = 0
    for c in centroids:
        clusters[c] = counter
        counter = counter + 1

    # apply algorithm
    for steps in range(n_iterations):

        for i in idx:
            rbo_dist = []
            for c in centroids:
                a = res_tmp[i]
                b = res_tmp[c]
                rbo_dist.append(rbo.rbo_ext(a, b, 0.9))
            
            clusters[i] = rbo_dist.index(min(rbo_dist))
    
    df = pd.DataFrame(
        {
            'index': [elem for elem in range(len(lol))],
            'cluster': clusters,
            'mean': np.repeat
        }
    )

    return df


def update_centroids(self, df, res):
    #empty list for new centroids
    centroids = np.repeat(None, k)
    #iterate over clusters
    for k in df.cluster.unique():
        #select lists for cluster
        cluster = df[df.cluster == k]
        for i in cluster.index:
            tmp = []
            #compute rbo to all other lists in the same cluster
            for j in cluster.index:
                tmp.append(rbo_ext(res[i], res[j], 0.9))
            #save mean of rbo to other results in cluster
            cluster.mean[i] = np.mean(tmp)
        centroids[k] = cluster.idxmin(axis=0)[2]
    return centroids
            
                


    







#%%
