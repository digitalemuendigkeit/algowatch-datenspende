#%% import packages
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import json
from helpers import get_domains

#%% import dataset
kwMetadata = pd.read_feather('analysis/data/2017-07-15/kwMetadata_SPD.feather')

# %% test significance of search time for clustering
# import statsmodels
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

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
pass