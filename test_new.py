#%%

# import packages
import json
import os.path
import importlib
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

# import for k-means clustering
import clustering_utilities as cu
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

#%%

# prepare
fdp = res['FDP'].copy()
rbo_mat = cu.compute_rbo_matrix(fdp, 0.9)
df = pd.DataFrame(rbo_mat)

# %%

# conduct elbow method
cu.plot_elbow(11, df)

#%%

# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
# conduct k-means clustering analysis
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=3000, n_init=10, random_state=0)
pred = kmeans.fit(df)

#%%

clus = cu.create_clus_output(fdp, pred.labels_)
# clus.to_csv('fdp.csv', sep=';')

#%%

# reduce to two dimensions
embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(df)

#%%

# format 2d data
xt = pd.DataFrame(X_transformed, columns=['dim1', 'dim2'])
xt = pd.concat([xt, pd.DataFrame(pred.labels_, columns=['cluster'])], axis=1)

plt.scatter(x=xt.dim1, y=xt.dim2, c=xt.cluster, alpha=0.3)
plt.show()

#%%

# reduce to three dimensions
embedding = MDS(n_components=3)
X_transformed = embedding.fit_transform(df)
xt = pd.DataFrame(X_transformed, columns=['dim1', 'dim2', 'dim3'])
xt = pd.concat([xt, pd.DataFrame(pred.labels_, columns=['cluster'])], axis=1)

#%%

# 3d visualization with plotly
import plotly.express as px
fig = px.scatter_3d(xt, x='dim1', y='dim2', z='dim3',
              color='cluster', opacity=0.3)

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

#%%

flat_list = [item for sublist in fdp for item in sublist]
flat_set = set(flat_list)
df_new = pd.DataFrame(flat_list, columns=['url'])
df_new = df_new[df_new.url != 'null']
counts = df_new['url'].value_counts()

#%%

importlib.reload(cu)
