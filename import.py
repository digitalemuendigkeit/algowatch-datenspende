# import json and os.path moduls
import json
import os.path
import pandas as pd
import numpy as np

# create system specific path to json file
path = os.path.join(
    'algowatch-datenspende' ,
    'data', 
    'datenspende_btw17_public_data_2017-09-29.json'
)

# read json file
with open(path, 'r') as json_file:
    json_data = json.load(json_file)

# separate result lists from meta data
result_lists = json_data[-1] 
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

# filter meta data frame
meta_data_df = df[df.search_type == "news"]
meta_data_df = meta_data_df.reset_index()
meta_data_df = meta_data_df.drop(
    ['plugin_id', 'index'], axis=1
)

