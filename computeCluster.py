
import os
import importlib
import pandas as pd
import numpy as np
import feather
import matplotlib.pyplot as plt
import multiprocessing as mp
import helpers
import re


def cluster_df(res, meta_data_df, kw, date):
    print("Computing clusters for " + kw)
    clus = helpers.apply_kmeans(res, kw)
    print("Done computing clusters for " + kw)
    kw_meta_df = helpers.get_kw_df(meta_data_df, clus, kw)
    if 'data' not in os.listdir('analysis'):
        os.mkdir(os.path.join('analysis', 'data'))
    if date not in os.listdir(os.path('analysis', 'data')):
        os.mkdir(os.path.join('analysis', 'data', date))
    # remove backslash for Buendnis90
    path = os.path.join(
        'analysis', 
        'data',
        date,
        'kwMetadata_' 
        + kw.replace("/", "_").replace("\\", "_").replace(" ", "_") 
        + '.feather'
    )
    feather.write_dataframe(kw_meta_df, path)
    return kw_meta_df

if __name__ == '__main__':
    datasets = []
    for file in os.listdir(os.path.join('Datasets')):
        if file.endswith(".json"):
            datasets.append(file)

    results = []
    for dataset in datasets:
        pool = mp.Pool(mp.cpu_count()-1)
        path = os.path.join('Datasets', dataset)
        res, meta_data_df = helpers.read_json(path)
        date = re.findall('\d{4}-\d{2}-\d{2}', dataset)[0]
        for kw in meta_data_df.keyword.unique():
            results = pool.apply_async(
                cluster_df,  
                args=(res, meta_data_df, kw, str(date))
            )         
        pool.close()
        pool.join()


