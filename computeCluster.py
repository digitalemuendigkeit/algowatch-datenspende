
import os.path
import importlib
import pandas as pd
import numpy as np
import feather
import matplotlib.pyplot as plt
import multiprocessing as mp
from progress.bar import Bar
import helpers


def cluster_df(res, meta_data_df, kw):
    print("Computing clusters for " + kw)
    clus = helpers.apply_kmeans(res, kw)
    print("Done computing clusters for " + kw)
    kw_meta_df = helpers.get_kw_df(meta_data_df, clus, kw)
    path = 'workingData/kwMetadata_' + kw + '.feather'
    feather.write_dataframe(kw_meta_df, path)
    return kw_meta_df

def merge_result(result):
    # global all_kw
    # all_kw = pd.merge(all_kw, result)
    global results
    results.append(result)

if __name__ == '__main__':
    datasets =[]
    for file in os.listdir(os.path.join('Datasets')):
        if file.endswith(".json"):
            datasets.append(file)

    results = []
    for dataset in datasets:
        pool = mp.Pool(mp.cpu_count()-1)
        path = os.path.join('Datasets', dataset)
        res, meta_data_df = helpers.read_json(path)
        # progress bar as this might take some time
        bar = Bar('Processing', max=len(meta_data_df.keyword.unique()))
        for kw in meta_data_df.keyword.unique():
            results = pool.apply_async(cluster_df,  args=(res, meta_data_df, kw))
            bar.next()          
        pool.join()
        pool.close()
        bar.finish() 

