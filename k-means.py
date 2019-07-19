#%%
import rbo
import numpy as np
import pandas as pd

#%%

def k_ls_rbo(lol, k, n_iterations):
    """k-means algorithm adjusted for list clustering"""

    # choose k centroids
    centroids = np.random.randint(0, len(lol), k)
    
    # create an index list and drop the indices of the centroids
    idx = [elem for elem in range(len(lol))]
    for k_idx in centroids:
        idx.pop(k_idx)
    
    # initialize clusters
    clusters = np.repeat('NA', len(lol))
    counter = 1
    for c in centroids:
        clusters[c] = counter
        counter = counter + 1

    # apply algorithm
    for steps in range(n_iterations):

        for i in idx:
            rbo_dist = []
            for c in centroids:
                rbo_dist.append(rbo.rbo_ext(res[i], res[c], 0.9))
            
            clusters[i] = rbo_dist.index(min(rbo_dist)) + 1
    
    df = pd.DataFrame(
        {'index': [elem for elem in range(len(lol))],
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
        controids[k] = cluster.idxmin(axis=0)[2]
    return centroids
            

                


    







#%%
