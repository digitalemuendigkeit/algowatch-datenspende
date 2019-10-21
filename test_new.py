#%%

# import packages
import json
import os.path
import pandas as pd
import numpy as np
import rbo
import matplotlib.pyplot as plt

#%%

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

fdp = res['FDP'].copy()


#%%

rbo_matrix = np.zeros(shape=(len(fdp), len(fdp)))

for i in range(len(fdp)):
    for j in range(i, len(fdp)):
        rbo_matrix[i][j] = rbo_matrix[j][i] = rbo.rbo_ext(fdp[i], fdp[j], p=0.9)

#%%

fdp_df = pd.DataFrame(rbo_matrix)

#%%

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%%

# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(fdp_df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#%%

# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(fdp_df)

#%%
clus_data = pd.DataFrame(zip(fdp, pred_y), columns=['result_list', 'cluster'])
#%%
clus_data[clus_data.cluster == 0]
#%%
clus_data[clus_data.cluster == 1]