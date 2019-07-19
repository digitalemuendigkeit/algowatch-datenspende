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
                a = res_tmp(i)
                b = res_tmp(c)
                rbo_dist.append(rbo.rbo_ext(a, b, 0.9))
            
            clusters[i] = rbo_dist.index(min(rbo_dist))
    
    df = pd.DataFrame(
        {
            'index': [elem for elem in range(len(lol))],
            'cluster': clusters
        }
    )

    return df



                


    







#%%
