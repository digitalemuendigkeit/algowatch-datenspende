import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import glob
import collections
import datetime
from tqdm import tqdm
from helpers import *
from hilbertcurve.hilbertcurve import HilbertCurve

keyword = "AfD"

files = glob.glob('analysis/data/**/kwMetadata_'+keyword+'.pkl', recursive=True)

for day_index, file in tqdm(enumerate(files)):
    # import dataset
    df = pd.read_pickle(file)

    # remove null entries in results
    df["urls"] = df["urls"].apply(lambda x: list(filter(("null").__ne__, x)))

    # flattend_df = pd.DataFrame()
    test = []
    for _, row in df.iterrows():
        for idx, url in enumerate(row.urls):
            new = row.copy()
            new["url"] = url
            new["rank"] = idx
            # flattend_df = flattend_df.append(new)
            test.append(new)

    flattend_df = pd.DataFrame(test)

    flattend_df.drop(['urls'], axis=1, inplace=True)
    flattend_df.drop(['index'], axis=1, inplace=True)
    flattend_df.drop(['result_hash'], axis=1, inplace=True)
    # trim urls
    flattend_df["url"] = flattend_df["url"].apply(lambda x: trim_url(x))
    flattend_df["url"] = flattend_df["url"].apply(lambda x: strip_google_link(x))
    # remove emty strings
    flattend_df = flattend_df[flattend_df["url"] != '']
    flattend_df.reset_index(inplace=True)
    # get domains
    flattend_df["domain"] = flattend_df["url"].apply(lambda x: get_domain_from_url(x))

    flattend_df.to_csv("workingData/preprocessed/"+keyword+"_"+str(day_index).zfill(2)+".csv")