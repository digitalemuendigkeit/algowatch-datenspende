#%%
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
from helpers import *
from hilbertcurve.hilbertcurve import HilbertCurve

def plot_hilbert_curve(keyword):
    # get all pickle files for a keyword
    files = glob.glob('analysis/data/**/kwMetadata_'+keyword+'.pkl', recursive=True)


    # %%
    # get previously computed urls
    with open("workingData/all_urls.pkl", "rb") as fp:
        all_urls = pickle.load(fp)

    # remove protocoll and www for sorting
    trimmed = list(map(trim_url, all_urls))
    #  remove empty strings
    trimmed = list(filter(len, trimmed))

    # %% get hilbert curve and map urls to number
    # we have 2**(N*p) numbers on the curve
    N=2 # 2 dimensions
    # map urls to numbers
    url_map, p = map_all_urls(trimmed, N)
    hilbert_curve = HilbertCurve(p, N)

    # get indices of of first url starting with a letter
    letter_indices = get_letter_indices(url_map)

    # %%
    # iterate over all results
    x, y, c, pos = ([] for i in range(4))
    for day_index, file in enumerate(files):
        # import dataset
        df = pd.read_pickle(file)
        # remove null entries in results
        df["urls"] = df["urls"].apply(lambda x: list(filter(("null").__ne__, x)))
        # trim urls
        df["urls"] = df["urls"].apply(lambda x: [trim_url(url) for url in x])
        # remove emty strings
        df["urls"] = df["urls"].apply(lambda x: list(filter(len, x)))
        # get domains
        df["domains"] = df["urls"].apply(lambda x: [get_domain_from_url(url) for url in x])
        # drop rows where length of results < 10
        df.drop(df[df["urls"].map(len) < 10].index, inplace=True)
        df.reset_index(inplace=True)
        # get flat list of domains
        domains =[domain for domains in list(df.domains) for domain in domains]
        # get most frequent domains
        top_domains = [d[0] for d in collections.Counter(domains).most_common(10)]
        # get indices for top domains
        domain_indices = get_domain_indices(url_map, top_domains)
        #get hilbert curve for day
        curve = np.zeros(((len(df)*10,4)))
        for idx in range(len(df)):
            for jdx in range(10):
                x_coord,y_coord = hilbert_curve.coordinates_from_distance(url_map[df.urls[idx][jdx]])
                curve[idx*10+jdx, 0] = x_coord
                curve[idx*10+jdx, 1] = y_coord
                curve[idx*10+jdx, 2] = df.cluster[idx]
                curve[idx*10+jdx, 3] = 50/(jdx+1)
        x_day, y_day, c_day, pos_day = curve.T
        x.append(x_day)
        y.append(y_day)
        c.append(c_day)
        pos.append(pos_day)
        plt.scatter(x_day, y_day, c=c_day, s=pos_day, alpha=0.3)
        plt.ylim(0,hilbert_curve.max_x)
        plt.xlim(0, hilbert_curve.max_x)
        # for l, idx in letter_indices.items():
        #     plt.annotate(l, # this is the text
        #          ( hilbert_curve.coordinates_from_distance(idx)), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center') # horizontal alignment can be left, right or center
        # insert annotations for most common domains
        for d, idx in domain_indices.items():
            plt.annotate(d, # this is the text
                (hilbert_curve.coordinates_from_distance(idx)), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,5), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
        date = datetime.datetime.utcfromtimestamp(df.timestamp[0]).strftime('%Y-%m-%d')
        plt.title(keyword + "    " + date)
        plt.savefig("analysis/plots/hilbert_"+keyword+"_"+str(day_index).zfill(2)+".png")
        plt.clf()
    return x, y, c, pos

def merge_to_video(keyword):
    command = "ffmpeg -r 3 -f image2 -s 640x480 -i analysis\\plots\\hilbert_"+keyword + "_%02d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p analysis\\videos\\" + keyword +".mp4"
    os.system(command)

if __name__ == "__main__":
    # keywords = [
    # 'Katrin_Göring-Eckardt', 'Christian_Lindner', 'FDP', 'AfD',
    # 'Alexander_Gauland', 'Martin_Schulz', 'SPD', 'Sahra_Wagenknecht',
    # 'Alice_Weidel', 'CDU', 'Die_Linke', 'CSU', 'Angela_Merkel',
    # 'Dietmar_Bartsch', 'Cem_Özdemir', 'Bündnis90_Die_Grünen'
    # ]
    keywords = [
    'AfD'
    ]
    for keyword in keywords:
        plot_hilbert_curve(keyword)
        merge_to_video(keyword)


# convert to video
# http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# ffmpeg -r 2 -f image2 -s 640x480 -i hilbert_AfD_%02d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
