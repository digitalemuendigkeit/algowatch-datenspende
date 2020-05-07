#%% import packages
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
from helpers import get_domains, get_full_url, map_all_urls, append_urls, map_urls
from hilbertcurve.hilbertcurve import HilbertCurve


#%% import dataset
kwMetadata = pd.read_feather('analysis/data/2017-07-15/kwMetadata_SPD.feather')

#%% get results
# read json file for results
path= "Datasets/datenspende_btw17_public_data_2017-07-15.json"
# read json file
with open(path, 'r') as json_file:
    json_data = json.load(json_file)
results = json_data[-1]
# create hashmap
hashmap = dict()
for result in results:
    tmp = dict(result)
    value = tmp['result']
    key = tmp['result_hash']
    hashmap[key] = value

# %%
# get result domains into kw df
kwMetadata["domains"] = kwMetadata["result_hash"].apply(lambda x: get_domains(hashmap[x]))

#%% get full url
kwMetadata["urls"] = kwMetadata["result_hash"].apply(lambda x: get_full_url(hashmap[x]))

#%% save feather
import feather
feather.write_dataframe(kwMetadata, "example.feather")
# %% test significance of search time for clustering

# test normality
stat, shapiro_p = stats.shapiro(kwMetadata['timestamp'])
print('Shapori statistics = %.3f, p = %.3f' % (stat, shapiro_p))

# test homogeneity  of variances
groups = []
gb = kwMetadata.groupby('cluster')
[groups.append(gb.get_group(x)['timestamp']) for x in gb.groups]
stat, levene_p = stats.levene(*groups)
print('Levene statistics = %.3f, p = %.3f' % (stat, levene_p))

# only conduct anova if assumptions are met
if(shapiro_p < 0.5 and levene_p > 0.5):
    # anova = ols('timestamp ~ C(cluster)', data=kwMetadata).fit()
    # anova.summary()
    # table = sm.stats.anova_lm(anova, typ=2)
    stat, p = stats.f_oneway(*groups)
else:
    stat, p = stats.kruskal(*groups)

print('Anova/Kruskal statistics = %.3f, p = %.3f' % (stat, p))


# %% plot boxplot for time stamps
# from matplotlib import dates

# d = groups[:,0]
# s = d/1000
# dts = map(datetime.datetime.fromtimestamp, s)
# fds = dates.date2num(dts) # converted

# matplotlib date format object
# hfmt = dates.DateFormatter('%H:%M')
fig, ax = plt.subplots()
ax.boxplot(groups)
# ax.yaxis.set_major_locator(dates.MinuteLocator())
# ax.yaxis.set_major_formatter(hfmt)
plt.show()


# %% Show frequency of languages
kwMetadata.groupby(['cluster','country']).count()
kwMetadata[(kwMetadata.country=="DE") & (kwMetadata.cluster==2)].groupby('language').count()

for idx in range(0, kwMetadata.cluster.max() + 1):
    german = kwMetadata[(kwMetadata.country=="DE") & (kwMetadata.cluster==idx)]['keyword'].count()
    total = kwMetadata[(kwMetadata.cluster==idx)]['keyword'].count()
    print('Share searches from DE in cluster %d: %.2f %% (N=%d)' % (idx, (german / total * 100), total))

# %% Scatterplot of cluster over time
fig, ax = plt.subplots()
ax.scatter(x=kwMetadata.timestamp, y=kwMetadata.cluster, c= kwMetadata.cluster, alpha= 0.4)






# %% plot clusters on hilber curve
curve = np.zeros(((len(kwMetadata)*10,3)))
for idx in range(len(kwMetadata)):
    for jdx in range(10):
        x,y = hilbert_curve.coordinates_from_distance(url_map[kwMetadata.urls[idx][jdx]])
        curve[idx*10+jdx, 0] = x
        curve[idx*10+jdx, 1] = y
        curve[idx*10+jdx, 2] = kwMetadata.cluster[idx]

# %%
x,y,c = curve.T
plt.scatter(x,y, c=c, alpha=0.3)
plt.savefig("hilbert_spd.png")
plt.show()

# %%
sns.scatterplot(x=x, y=y,hue=c, alpha=0.3, x_jitter=True, y_jitter=True, palette="bright")

# %%
# get all pickle files for a keyword
import glob
keyword = "AfD"
files = glob.glob('analysis/data/**/kwMetadata_'+keyword+'.pkl', recursive=True)


# %%
# get previously computed urls
with open("workingData/all_urls.pkl", "rb") as fp:
    all_urls = pickle.load(fp)

#%% 

#%% get hilbert curve and map urls to number
# we have 2**(N*p) numbers on the curve
N=2 # 2 dimensions
# map urls to numbers
url_map, p = map_all_urls(all_urls, N)
hilbert_curve = HilbertCurve(p, N)

# %%
# iterate over all results
x = y = c = d = []
for day, file in enumerate(files[:5]):
    # import dataset
    kwMetadata = pd.read_pickle(file)
    # get full url
    # kwMetadata["urls"] = kwMetadata["result_hash"].apply(lambda x: get_full_url(hashmap[x]))
    #get hilbert curve for day
    curve = np.zeros(((len(kwMetadata)*10,4)))
    for idx in range(len(kwMetadata)):
        for jdx in range(10):
            x,y = hilbert_curve.coordinates_from_distance(url_map[kwMetadata.urls[idx][jdx]])
            curve[idx*10+jdx, 0] = x
            curve[idx*10+jdx, 1] = y
            curve[idx*10+jdx, 2] = kwMetadata.cluster[idx]
            curve[idx*10+jdx, 3] = day
    x_day, y_day, c_day, day = curve
    x.append(x_day)
    y.append(y_day)
    c.append(c_day)
    d.append(day)


# %% get ALL urls from all datasets
datasets = []
for file in os.listdir(os.path.join('Datasets')):
    if file.endswith(".json"):
        datasets.append(file)
all_urls = []
for dataset in datasets:
    with open("Datasets/"+ dataset, 'r') as json_file:
        json_data = json.load(json_file)
    results = json_data[-1]
    # create hashmap
    hashmap = dict()
    # for result in results:
    #     tmp = dict(result)
    #     value = tmp['result']
    #     key = tmp['result_hash']
    #     hashmap[key] = value
    all_urls = append_urls(results, all_urls)
