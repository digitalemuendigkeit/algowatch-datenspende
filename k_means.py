#%%

import rbo
import numpy as np
import pandas as pd

#%%

def k_means_rbo(lol, k, max_iter, exit_th = 0.9):
    """k-means algorithm adjusted for list clustering"""

    # preparation
    res_tmp = lol

    # choose k centroids at random (initialization)
    centroids = np.random.choice(len(res_tmp), k, replace=False)
    centroids.sort()
    
    # initialize clusters
    clusters = np.repeat(-1, len(res_tmp))
    counter = 0
    for c in centroids:
        clusters[c] = counter
        counter = counter + 1

    # initialize data frame
    df = pd.DataFrame(
        {
            'elem_index': [elem for elem in range(len(res_tmp))],
            'cluster': clusters,
            'mean_rbo': np.repeat(0.0, len(res_tmp))
        }
    )

    # k-means iterations
    for steps in range(max_iter):
        
        # test print
        print(centroids)

        # create an index list and drop the indices of the centroids
        idx = [elem for elem in range(len(res_tmp))]
        drop_counter = 0
        iter_centroids = centroids
        iter_centroids.sort()
        for k_idx in iter_centroids:
            idx.pop(k_idx - drop_counter)
            drop_counter = drop_counter + 1

        # iterate over index list
        for i in idx:
            rbo_dist = []
            for c in centroids:
                rbo_dist.append(rbo.rbo_ext(res_tmp[i], res_tmp[c], 0.9))
            df.iloc[i, 1] = rbo_dist.index(max(rbo_dist))
        tmp = update_centroids(df, res_tmp, k)
        if is_over_exit_thresh(res_tmp, centroids, tmp, exit_th):
            print("passed exit threshold")
            break
        centroids = tmp
    
    # drop mean_rbo column
    df = df.drop(columns = 'mean_rbo')

    return df, centroids, compute_distortion(res_tmp, df)

#%%

def update_centroids(df, res, n_clust):
    """update centroids after k-means iteration"""
    
    # empty list for new centroids
    centroids_updated = np.repeat(-1, n_clust)
    
    # iterate over clusters
    for clust in df.cluster.unique():
        
        # subset data frame for cluster k
        cluster = df[df.cluster == clust]
        cluster.set_index('elem_index', drop=False, inplace=True)
        
        # iterate over cluster k
        for i in cluster.elem_index:
            
            # initialize temporary list for mean rbo scores
            tmp = []
           
            # compute rbo to all other lists in the same cluster
            for j in cluster.elem_index:
                tmp.append(rbo.rbo_ext(res[i], res[j], 0.9))
            
            # save mean of rbo to other results in cluster
            cluster.at[i, 'mean_rbo'] = np.mean(tmp)

        # update new centroids list        
        centroids_updated[clust] = cluster.idxmax(axis=0)['mean_rbo']
        
    return centroids_updated

#%%

def is_over_exit_thresh(res, centroids, centroids_new, exit_thresh):
    """test for exit criterium"""

    # initiate list containing rbo comparison between last two centroid sets
    thresh_list = []

    # iterate over centroid sets
    for i in range(len(centroids)):
        thresh_list.append(rbo.rbo_ext(res[centroids[i]], res[centroids_new[i]], 0.9))
    
    # if all rbo scores between last two centroids are over threshold, return True
    if all(elem > exit_thresh for elem in thresh_list):
        return True
    else:
        return False

#%%

def compute_distortion(res, df):
    """compute distortion within clusters for elbow method"""
    
    distortions = pd.DataFrame(
        {
            'cluster': np.array([]),
            'sum_rbo': np.array([]),
            'n_clust': np.array([])
        }
    )

    # iterate over clusters
    for clust in df.cluster.unique():
        
        # subset data frame for cluster k
        cluster = df[df.cluster == clust]
        cluster.set_index('elem_index', drop=False, inplace=True)
        
        # iterate over cluster k
        for i in cluster.elem_index:
            
            # initialize temporary list for mean rbo scores
            tmp = []
           
            # compute rbo to all other lists in the same cluster
            for j in cluster.elem_index:
                tmp.append(rbo.rbo_ext(res[i], res[j], 0.9))
            
            # save mean of rbo to other results in cluster
            cluster.at[i, 'mean_rbo'] = np.mean(tmp)

        # create temporary data frame with current distortion
        tmp = pd.DataFrame(
            {
                'cluster': np.array([clust]),
                'sum_rbo': np.mean(cluster['mean_rbo'].values),
                'n_clust': np.array([cluster.shape[0] / df.shape[0]])
            }
        )

        # append to distortions data frame
        distortions = pd.concat([distortions, tmp])
        
    return distortions

