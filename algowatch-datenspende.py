# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#%%

# import packages
import json
import os.path
import pandas as pd
import numpy as np
import k_means
import matplotlib.pyplot as plt

#%%

# import json file

# create system specific path to json file
path = os.path.join(
    'data', 
    'datenspende_btw17_public_data_2017-09-29.json'
)

# read json file
with open(path, 'r') as json_file:
    json_data = json.load(json_file)

#%%

# separate result lists from meta data
results = json_data[-1] 
meta_data = json_data[0:(len(json_data)-1)]

# create data frame with meta data
df = pd.DataFrame(meta_data[0])
df['person_id'] = pd.Series(
    np.repeat(np.array([1]), df.shape[0], axis=0),
    index=df.index
)
for i in range(1, len(meta_data)-1):
    tmp = pd.DataFrame(meta_data[i])
    tmp['person_id'] = pd.Series(
        np.repeat(np.array([i]), tmp.shape[0], axis=0),
        index=tmp.index
    )
    df = pd.concat([df, tmp])

#%%

# filter meta data frame
meta_data_df = df[df.search_type == "search"]
meta_data_df = meta_data_df.reset_index()
meta_data_df = meta_data_df.drop(
    ['plugin_id', 'index', 'plugin_version'], axis=1
)

#%%

# create hashmap
hashmap = dict()
for result in results:
    tmp = dict(result)
    value = tmp['result']
    key = tmp['result_hash']
    hashmap[key] = value

#%%

# create result list

# extract keywords
keywords = meta_data_df.keyword.unique()

# initialize empty list
result_lists = [None] * (len(keywords))

for idx, keyword in enumerate(keywords):
    tmp = []
    for _, search in meta_data_df[meta_data_df.keyword == keyword].iterrows():
        #check whether results are empty before appending
        if(hashmap[search.result_hash] is not None):
            tmp.append(hashmap[search.result_hash])
    result_lists[idx] = tmp

#%%

# add names to result lists
result_lists = dict(zip(keywords, result_lists))

#%%

# initialize dictionary with keywords as keys
res = dict((kw, []) for kw in keywords)

# unpack result_lists into res
for kw in keywords:
    for i in range(len(result_lists[kw])):
        tmp = []
        for j in range(len(result_lists[kw][i])):
            tmp.append(result_lists[kw][i][j]["sourceUrl"])
        res[kw].append(tmp)

#%%

# subset res to create test set
res_test = res['FDP'][0:100]

# apply k_means_rbo on res_test
clus, centr, distort = k_means.k_means_rbo(res_test, 5, 20, 0.95)

#%%

# subset res to create a second test set
res_test2 = res['FDP'][0:100]

# set maximum k for elbow criterium
max_k = 10

# initialize elbow runs
distort_list = []
k_vals = list(range(1, 11, 1))

# run algorithm for each k up to max_k (10 iterations each)
for k_iter in range(1, max_k+1, 1):
    k_iter_distort_list = []
    for i in range(10):
        tmp_clus, tmp_centr, tmp_distort = k_means.k_means_rbo(
            res_test2, k_iter, 10, 0.95
        )
        k_iter_distort_list.append(
            (tmp_distort['mean_rbo'] * tmp_distort['cluster_weight']).sum()
        )
    distort_list.append(max(k_iter_distort_list))

# plot results
plt.plot(k_vals, distort_list, 'bo')
plt.show()

#%%

# subset res to create test set
res_test3 = res['FDP']

# apply k_means_rbo on res_test
clus3, centr3, distort3 = k_means.k_means_rbo(res_test3, 5, 20, 0.95)

#%%

# for reloading in case of changes
from importlib import reload
reload(k_means)


#%%
