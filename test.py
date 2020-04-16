#%%
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import json
from helpers import get_domains, get_full_url, map_all_urls
from hilbertcurve.hilbertcurve import HilbertCurve

#%% read json file for results
path= "Datasets/datenspende_btw17_public_data_2017-07-15.json"
# read json file
with open(path, 'r') as json_file:
    json_data = json.load(json_file)
results = json_data[-1]

url_map = map_all_urls(results)

#%%
# we have 2**(N*p) numbers on the curve
N=2 # 2 dimensions
p= 7 #2**(2*7) =16384 > 10943 = len(urls) 
hilbert_curve = HilbertCurve(p, N)



# %%
curve = np.zeros(((len(url_map),3)))
for idx, url in enumerate(url_map.keys()):
    x,y = hilbert_curve.coordinates_from_distance(url_map[url])
    curve[idx, 0] = x
    curve[idx, 1] = y


# %%
x,y,_ = curve.T
plt.scatter(x,y)
plt.show()



# %%
