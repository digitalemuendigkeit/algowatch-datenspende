# import packages
import json
import os.path
import importlib
import pandas as pd
import numpy as np
import math
import rbo
import datetime
import re
# import for k-means clustering
import clustering_utilities as cu
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

# wrapper method to apply timestamp method to column
def get_timestamp(x):
   return x.timestamp()

def get_country(x):
   return x.split()[0]

def get_domains(results):
   expression = r"(?:https?:\/\/)?(?:www\.)?([^\/\r\n]+)(?:\/[^\r\n]*)?"
   domains = []
   for result in results:
      url = result["sourceUrl"]
      domain = re.findall(expression, url)[0]
      domains.append(domain)
   return domains

def get_full_url(results):
   urls = []
   for result in results:
      url = result["sourceUrl"]
      urls.append(url)
   return urls
# reads json dataset from path and returns result list and metadata DF
def read_json(path):
   # read json file
   with open(path, 'r') as json_file:
      json_data = json.load(json_file)

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

   # filter meta data frame
   meta_data_df = df[df.search_type == "search"]
   meta_data_df = meta_data_df.reset_index()

   for c in ['plugin_id', 'index', 'plugin_version']:
      try:
         meta_data_df = meta_data_df.drop(c, axis=1)
      except:
         continue

   # create hashmap
   hashmap = dict()
   for result in results:
      tmp = dict(result)
      value = tmp['result']
      key = tmp['result_hash']
      hashmap[key] = value

   # extract keywords
   keywords = meta_data_df.keyword.unique()

   #initialize colum vor urls
   meta_data_df["urls"] = ""

   # initialize empty list
   # result_lists = [None] * (len(keywords))
   result_lists = []

   for idx, keyword in enumerate(keywords):
      tmp = []
      #TODO: delete rows with empty results
      for jdx, search in meta_data_df[meta_data_df.keyword == keyword].iterrows():
         #check whether results are empty before appending
         if(hashmap[search.result_hash] is not None):
            urls = []
            for result in hashmap[search.result_hash]:
               urls.append(result["sourceUrl"])
            if len(urls)>0:
               tmp.append(urls)
               meta_data_df["urls"][jdx] = urls
            else:
               meta_data_df = meta_data_df.drop(jdx)
      result_lists.append(tmp)

   meta_data_df = meta_data_df.reset_index()

   # add names to result lists
   res = dict(zip(keywords, result_lists))

   # initialize dictionary with keywords as keys
   # res = dict((kw, []) for kw in keywords)

   # unpack result_lists into res
   # for kw in keywords:
   #    for i in range(len(result_lists[kw])):
   #       tmp = []
   #       for j in range(len(result_lists[kw][i])):
   #             tmp.append(result_lists[kw][i][j]["sourceUrl"])
   #       res[kw].append(tmp)

   return res, meta_data_df

def apply_kmeans(res, keyword):
   # compute RBO matrix for keyword
   kw_df= res[keyword].copy()
   rbo_mat = cu.compute_rbo_matrix(kw_df, 0.9)
   df = pd.DataFrame(rbo_mat)

   # conduct elbow method
   wcss = cu.plot_elbow(10, df, False)
   n_clusters = cu.optimal_number_of_clusters(10, wcss)

   # https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
   # conduct k-means clustering analysis
   kmeans = KMeans(n_clusters= n_clusters, init='k-means++', max_iter=3000, n_init=10, random_state=0)
   pred = kmeans.fit(df)

   clus = cu.create_clus_output(kw_df, pred.labels_)
   return clus

def get_kw_df(meta_data_df, clus, keyword):   
   kwMetadata = meta_data_df[meta_data_df['keyword']==keyword]
   kwMetadata = kwMetadata.reset_index(drop=True)
   kwMetadata = pd.merge(kwMetadata, clus['cluster'], left_index=True, right_index=True)
   # convert search_date to timestamp
   kwMetadata['search_date'] = kwMetadata['search_date'].apply(datetime.datetime.strptime, args=['%Y-%m-%d %H:%M'])
   kwMetadata['timestamp'] = kwMetadata['search_date'].apply(get_timestamp)
   # extract country
   kwMetadata['country'] = kwMetadata['geo_location'].apply(get_country)
   return kwMetadata

def map_all_urls(urls, n=2):
   """
      urls: list of all results
      returns a dict of all urls mapped to an integer value, scaled to the clostest
      power of 2 in order to be mapped on the hilber curve
   """
   # urls= []
   # # iterate over all result lists
   # for result in results:
   #    # iterate over all entries in one result list
   #    for entry in result["result"]:
   #       urls.append(entry["sourceUrl"])
   # # remove duplicates
   # urls = list(set(urls))
   # sort
   urls.sort()
   #find closest *even* power of 2 to scale
   hilbert_length = math.ceil(np.log2(len(urls)))
   hilbert_length += hilbert_length % 2
   scale = 2**hilbert_length / len(urls)
   p= int(hilbert_length / n)
   # get dict with url as key and index as value
   url_map = {}
   for idx, url in enumerate(urls):
      url_map[url] = int(idx*scale)
   return url_map, p

def append_urls(results, urls):
   # iterate over all result lists
   for result in results:
      # iterate over all entries in one result list
      for entry in result["result"]:
         urls.append(entry["sourceUrl"])
   # remove duplicates
   urls = list(set(urls))
   return urls
 
def map_urls(urls):
   """
      urls: list of ulrs strings 
      returns a dict of all urls mapped to an integer value, scaled to the clostest
      power of 2 in order to be mapped on the hilber curve
   """
   # remove duplicates
   urls = list(set(urls))
   # sort
   urls.sort()
   #find closest power of 2 to scale
   hilbert_length = math.ceil(np.log2(len(urls)))
   scale = 2**hilbert_length / len(urls)
   # get dict with url as key and index as value
   url_map = {}
   for idx, url in enumerate(urls):
      url_map[url] = int(idx*scale)
   return url_map