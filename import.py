# import json and os.path moduls
import json
import os.path

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
meta_data = json_data[0:(len(json_data) - 1)]

